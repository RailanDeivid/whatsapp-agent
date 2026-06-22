import asyncio
import base64
import io
import logging
import time
import uuid

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import redis as redis_lib
from langchain.tools import BaseTool

from src.connectors.dremio import client as dremio_client
from src.connectors.mysql import client as mysql_client
from src.config import REDIS_URL
from src.tools.utils import extract_json

logger = logging.getLogger(__name__)

_CHART_KEY_PREFIX = "chart:"
_CHART_TTL = 120  # segundos

_redis = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, max_connections=10)

_THEME = {
    "primary":   "#1a6b2f",
    "accent":    "#2ecc71",
    "bg":        "#ffffff",
    "grid":      "#eeeeee",
    "text_dark": "#1a1a1a",
    "text_mid":  "#555555",
    "font":      "DejaVu Sans",
    "dpi":       160,
}

_PRIMARY    = _THEME["primary"]
_ACCENT     = _THEME["accent"]
_BG         = _THEME["bg"]
_GRID       = _THEME["grid"]
_TEXT_DARK  = _THEME["text_dark"]
_TEXT_MID   = _THEME["text_mid"]

_matplotlib_ready = False


def _setup_matplotlib() -> None:
    """Configura o backend não-interativo uma única vez (evita side-effect no import)."""
    global _matplotlib_ready
    if not _matplotlib_ready:
        matplotlib.use("Agg")
        _matplotlib_ready = True

def _green_gradient(n: int) -> list:
    cmap = plt.cm.get_cmap("Greens")
    return [cmap(0.32 + 0.58 * i / max(n - 1, 1)) for i in range(n)]

def _pie_palette(n: int) -> list:
    """Combina verdes com tons suaves complementares para legibilidade no pizza."""
    base = [
        "#27ae60", "#52b856", "#a8d5a2", "#145a27",
        "#7ec87a", "#1e8449", "#2ecc71", "#1a6b2f",
        "#b2dfdb", "#80cbc4",
    ]
    return (base * ((n // len(base)) + 1))[:n]

def _fmt(v: float) -> str:
    return "R$ {:,.0f}".format(v).replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_pct(pct: float) -> str:
    return f"{pct:.1f}%"

def _parse_title(titulo: str):
    """Divide 'Titulo Principal | Subtitulo' em (principal, subtitulo)."""
    parts = [p.strip() for p in titulo.split("|", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""

def _apply_base_style():
    _setup_matplotlib()
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update({
        "font.family": _THEME["font"],
        "axes.facecolor": _BG,
        "figure.facecolor": _BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#dddddd",
        "axes.linewidth": 0.8,
        "grid.color": _GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.color": _TEXT_MID,
        "ytick.color": _TEXT_MID,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

def _add_titles(fig, ax, titulo: str):
    main, sub = _parse_title(titulo)
    ax.set_title(main, fontsize=15, fontweight="bold", color=_TEXT_DARK,
                 pad=20 if not sub else 8, loc="center")
    if sub:
        ax.text(0.5, 1.03, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10,
                color=_TEXT_MID, style="italic")

def _add_footer_line(fig):
    fig.add_artist(plt.Line2D(
        [0.07, 0.93], [0.005, 0.005],
        transform=fig.transFigure,
        color=_PRIMARY, linewidth=2.5,
        solid_capstyle="round",
    ))

def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_THEME["dpi"], bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _build_bar_chart(df, titulo: str, col_categoria: str, col_valor: str) -> str:
    _apply_base_style()
    df = df[[col_categoria, col_valor]].dropna()
    df[col_valor] = df[col_valor].astype(float)
    df = df.sort_values(col_valor, ascending=True).reset_index(drop=True)

    cats = df[col_categoria].astype(str).tolist()
    vals = df[col_valor].tolist()
    n = len(cats)
    colors = _green_gradient(n)
    horizontal = n > 6

    fig_w = 10
    fig_h = max(6.0, n * 0.52) if horizontal else 6.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    if horizontal:
        bars = ax.barh(cats, vals, color=colors, edgecolor="white",
                       linewidth=0.8, height=0.62, zorder=3)
        for bar, val, color in zip(bars, vals, colors):
            ax.text(
                val + max(vals) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                _fmt(val), va="center", ha="left",
                fontsize=8.5, fontweight="bold", color=_TEXT_DARK,
            )
        ax.set_xlim(0, max(vals) * 1.28)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))
        ax.grid(axis="x", zorder=0)
        ax.set_xlabel("")
        ax.tick_params(axis="y", length=0)
    else:
        bars = ax.bar(cats, vals, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.58, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(vals) * 0.012,
                _fmt(val), ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=_TEXT_DARK,
            )
        ax.set_ylim(0, max(vals) * 1.24)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))
        ax.grid(axis="y", zorder=0)
        plt.xticks(rotation=20, ha="right")
        ax.tick_params(axis="x", length=0)

    _add_titles(fig, ax, titulo)
    _add_footer_line(fig)
    plt.tight_layout(pad=2.0)
    return _to_b64(fig)


def _build_line_chart(df, titulo: str, col_categoria: str, col_valor: str) -> str:
    _apply_base_style()
    df = df[[col_categoria, col_valor]].dropna()
    df[col_valor] = df[col_valor].astype(float)

    cats = df[col_categoria].astype(str).tolist()
    vals = df[col_valor].tolist()
    n = len(cats)
    x = list(range(n))

    fig_w = max(10, n * 0.65)
    fig_h = 6.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.fill_between(x, vals, alpha=0.12, color=_PRIMARY, zorder=1)
    ax.plot(x, vals, color=_ACCENT, linewidth=5, alpha=0.18, zorder=2,
            solid_capstyle="round")
    ax.plot(x, vals, color=_PRIMARY, linewidth=2.8, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(x, vals, color=_BG, edgecolors=_PRIMARY,
               s=80, linewidth=2.5, zorder=4)

    for xi, val in zip(x, vals):
        ax.text(xi, val + max(vals) * 0.035, _fmt(val),
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=_TEXT_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(
        cats,
        rotation=25 if n > 8 else 0,
        ha="right" if n > 8 else "center",
        fontsize=9,
    )
    ax.set_ylim(0, max(vals) * 1.28)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda xv, _: _fmt(xv)))
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", color="#f0f0f0", linestyle=":", linewidth=0.5, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    _add_titles(fig, ax, titulo)
    _add_footer_line(fig)
    plt.tight_layout(pad=2.0)
    return _to_b64(fig)


def _build_pie_chart(df, titulo: str, col_categoria: str, col_valor: str) -> str:
    _apply_base_style()
    df = df[[col_categoria, col_valor]].dropna()
    df[col_valor] = df[col_valor].astype(float)
    df = df.sort_values(col_valor, ascending=False).reset_index(drop=True)

    # Agrupa fatias menores que 2% em "Outros"
    total = df[col_valor].sum()
    mask = (df[col_valor] / total) < 0.02
    if mask.sum() > 1:
        outros_val = df.loc[mask, col_valor].sum()
        df = df[~mask].copy()
        df = df._append(
            {col_categoria: "Outros", col_valor: outros_val},
            ignore_index=True,
        )

    labels = df[col_categoria].astype(str).tolist()
    vals = df[col_valor].tolist()
    n = len(labels)
    colors = _pie_palette(n)

    # Destaca levemente a maior fatia
    explode = [0.04 if i == 0 else 0.0 for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        pctdistance=0.72,
        startangle=140,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 2.0},
        textprops={"fontsize": 9},
    )

    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
        at.set_color("white")

    legend_labels = [
        f"{lbl}  –  {_fmt(v)}  ({v/total*100:.1f}%)"
        for lbl, v in zip(labels, vals)
    ]
    handles = [mpatches.Patch(color=c) for c in colors]
    ax.legend(
        handles, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.5,
        frameon=False,
        labelcolor=_TEXT_DARK,
    )

    ax.text(0, 0, f"Total\n{_fmt(total)}", ha="center", va="center",
            fontsize=10, fontweight="bold", color=_TEXT_DARK)

    _add_titles(fig, ax, titulo)
    _add_footer_line(fig)
    plt.tight_layout(pad=2.0)
    return _to_b64(fig)


class ChartTool(BaseTool):
    name: str = "gerar_grafico"
    description: str = (
        "QUANDO USAR: SOMENTE quando o usuario pedir EXPLICITAMENTE um grafico, chart ou visualizacao. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: grafico, chart, mostre em grafico, quero ver em grafico, "
        "faz um grafico, me mostra visualmente, pizza, barra, linha (no contexto de grafico). "
        "NUNCA use para respostas normais de dados em texto — so quando o usuario pedir grafico explicitamente. "
        "Input: JSON com campos: "
        "'sql' (query SQL com EXATAMENTE 2 colunas: categoria e valor numerico agregado), "
        "'titulo' (OBRIGATORIO: use SEMPRE datas concretas — NUNCA 'Ontem', 'Hoje', 'Semana Passada' ou termos vagos. "
        "Formatos: dia unico → 'DD/MM/AAAA'; semana/periodo → 'DD/MM a DD/MM/AAAA'; mes → 'Nome do Mes AAAA'; ano → 'AAAA'. "
        "Exemplos: 'Vendas por Bar | 12/03/2026', 'Faturamento | 03/03 a 09/03/2026', 'Vendas por Categoria | Marco 2026', 'Faturamento | 2026'), "
        "'col_categoria' (nome exato da coluna de categorias), "
        "'col_valor' (nome exato da coluna de valores), "
        "'fonte' (OBRIGATORIO: 'dremio' para dados de vendas/faturamento/delivery/metas, 'mysql' para dados de compras/fornecedores), "
        "'tipo' (OPCIONAL: 'barra' [padrao], 'linha' para evolucao temporal, 'pizza' para participacao percentual). "
        "Use 'linha' quando o usuario pedir evolucao por dia/hora/periodo. "
        "Use 'pizza' quando o usuario pedir participacao, share, distribuicao percentual ou 'pizza'. "
        "Exemplo barra: {\"sql\": \"SELECT unidade, SUM(valor_liquido) AS total "
        "FROM views.\\\"ANALYTICS\\\".\\\"fTransacoes\\\" WHERE ... GROUP BY unidade\", "
        "\"titulo\": \"Vendas por Bar | 12/03/2026\", "
        "\"col_categoria\": \"unidade\", \"col_valor\": \"total\", \"fonte\": \"dremio\", \"tipo\": \"barra\"}. "
        "Exemplo pizza: {\"sql\": \"SELECT unidade, SUM(valor_liquido) AS total "
        "FROM views.\\\"ANALYTICS\\\".\\\"fTransacoes\\\" WHERE ... GROUP BY unidade\", "
        "\"titulo\": \"Participacao por Bar | Marco 2026\", "
        "\"col_categoria\": \"unidade\", \"col_valor\": \"total\", \"fonte\": \"dremio\", \"tipo\": \"pizza\"}. "
        "Exemplo linha: {\"sql\": \"SELECT data_evento, SUM(valor_liquido) AS total "
        "FROM views.\\\"ANALYTICS\\\".\\\"fTransacoes\\\" WHERE ... GROUP BY data_evento ORDER BY data_evento\", "
        "\"titulo\": \"Vendas por Dia | Unidade G BH | Marco 2026\", "
        "\"col_categoria\": \"data_evento\", \"col_valor\": \"total\", \"fonte\": \"dremio\", \"tipo\": \"linha\"}"
    )

    def _run(self, query: str) -> str:
        try:
            params = extract_json(query)
        except (ValueError, Exception) as e:
            return f"Erro: nao foi possivel interpretar o JSON do grafico. Detalhe: {e}"

        sql          = params.get("sql", "").strip()
        titulo       = params.get("titulo", "Grafico")
        col_categoria = params.get("col_categoria", "").strip()
        col_valor    = params.get("col_valor", "").strip()
        tipo         = params.get("tipo", "barra").strip().lower()
        fonte        = params.get("fonte", "dremio").strip().lower()

        if not sql or not col_categoria or not col_valor:
            return "Erro: campos 'sql', 'col_categoria' e 'col_valor' sao obrigatorios."

        logger.info("[grafico] Gerando [%s] '%s' (fonte=%s). SQL: %s", tipo, titulo, fonte, sql)
        t0 = time.time()
        try:
            df = mysql_client(sql) if fonte == "mysql" else dremio_client(sql)
        except Exception as e:
            logger.error("[grafico] Erro ao executar query apos %.1fs: %s", time.time() - t0, e)
            return f"Erro ao consultar dados para o grafico: {e}"

        if df.empty:
            logger.info("[grafico] Query retornou 0 linhas em %.1fs — grafico nao gerado.", time.time() - t0)
            return "Nenhum dado encontrado para gerar o grafico."

        logger.info("[grafico] %d linhas obtidas em %.1fs. Renderizando...", len(df), time.time() - t0)

        if col_categoria not in df.columns:
            return f"Erro: coluna '{col_categoria}' nao encontrada. Colunas: {list(df.columns)}"
        if col_valor not in df.columns:
            return f"Erro: coluna '{col_valor}' nao encontrada. Colunas: {list(df.columns)}"

        try:
            t1 = time.time()
            if tipo == "linha":
                b64 = _build_line_chart(df, titulo, col_categoria, col_valor)
            elif tipo in ("pizza", "pie"):
                b64 = _build_pie_chart(df, titulo, col_categoria, col_valor)
            else:
                b64 = _build_bar_chart(df, titulo, col_categoria, col_valor)
            logger.info("[grafico] Renderizado em %.1fs (tipo=%s, linhas=%d).", time.time() - t1, tipo, len(df))
        except Exception as e:
            logger.error("[grafico] Erro ao renderizar grafico tipo=%s titulo='%s': %s", tipo, titulo, e)
            return f"Erro ao renderizar o grafico: {e}"
        finally:
            plt.close("all")

        key = f"{_CHART_KEY_PREFIX}{uuid.uuid4().hex}"
        _redis.setex(key, _CHART_TTL, b64)
        logger.info("[grafico] Armazenado em Redis: key=%s | total=%.1fs", key, time.time() - t0)

        return f"[CHART:{key}|caption:{titulo}]"

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
