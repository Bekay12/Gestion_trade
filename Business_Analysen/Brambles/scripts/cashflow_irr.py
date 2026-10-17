#!/usr/bin/env python3
"""
cashflow_irr.py - Cash-flow schedules, IRR and sensitivity tables for n investment
options. Standard library only; deterministic output.

Generalised from a graded corporate-finance analysis. It answers section 6 of the
report template: per-option yearly cash-flow tables, a comparison table at the end
year, an IRR sensitivity across several horizons, and a margin sensitivity across
alternative revenue bases.

VIEW
    Incremental. The operating cash flow shown is the option's own contribution
    (earnings uplift minus the build-up of tied capital), not the group's total
    cash flow. This keeps the effect of the decision visible instead of drowning
    it in the base business. Declare this in the report as an explicit assumption.

USAGE
    python3 cashflow_irr.py --config analysis.json
    python3 cashflow_irr.py --config analysis.json --format latex --outdir data/
    python3 cashflow_irr.py --example > analysis.json

CONFIG SCHEMA (JSON)
{
  "currency": "Mio. EUR",
  "opening_liquidity": 138.2,       // cash at the start of horizon_start
  "horizon_start": 2026,
  "horizon_end": 2035,              // IRR horizon; MUST be stated in the report
  "schedule_years": [2026,2027,2028],   // years shown in the per-option tables
  "sensitivity_horizons": [2030, 2035, 2040],
  "revenue_base": 203.985,          // for the margin at the end of schedule_years
  "earnings_base": 27.278,          // operating EBITDA of the base year
  "revenue_scenarios": [            // alternative denominators; see rigor doc §3
    {"label": "base year held constant", "revenue": 203.985},
    {"label": "+3 % p.a.",               "growth": 0.03, "years": 3},
    {"label": "prior-year level",        "revenue": 312.346}
  ],
  "options": {
    "A": {
      "name": "Diversification",
      "investment":      {"2026": 11.7, "2027": 11.7, "2028": 11.6},
      "working_capital": {"2027": 7.5,  "2028": 7.5},
      "earnings_uplift": 16.0,
      "uplift_start": 2028,
      "terminal_value": 0,             // optional; see below
      "remarks": {"2026": "market development", "2027": "...", "2028": "..."}
    }
  }
}

"terminal_value" is what the position is still worth at the end of the horizon. It
is zero for a project whose spend is gone, and it is the whole point for an
ownership position: a listed share is entry price at t=0, distributable cash flow
per share each year, and the stake's value at the end. Omit it there and the model
answers a different question - how long the distributions alone need to repay the
price at the hurdle - whose break-even lands far below the market price for every
company on earth. Unlike "investment", it does NOT scale under
break_even_investment(): an exit value does not grow because the entry price did.
Source it (book equity per share is the defensible default) and declare it, because
it dominates the result. required_terminal_value() inverts the question and reports
what the CURRENT price already presupposes.

Two modelling traps the schema deliberately exposes:
  - "investment" is a per-year mapping, not a total. An acquisition price falls due
    at closing, not in equal thirds; splitting it evenly hides the liquidity trough.
  - "working_capital" is tied up in the year BEFORE the first uplift AND in the uplift
    year itself, because working capital pre-finances revenue and keeps growing through
    the ramp. See options A and B below: uplift_start 2028 with tie-up in 2027+2028,
    uplift_start 2027 with tie-up in 2026+2027. An acquisition is the exception - the
    acquired working capital transfers in full at closing (option C).

Whatever convention you choose, describe it in the report exactly as the config encodes
it. In the reference case the report said "in the year(s) immediately before the earnings
contribution" while the numbers tied up half the working capital in the contribution year
itself. The tables were right and the prose was wrong, which is the harder defect to find.

This model is PRE-TAX and pre-depreciation. "earnings_uplift" is an EBITDA figure: no tax
on it, no depreciation on the investment. The resulting IRR is therefore a pre-tax return
and must be published as one. See references/rigor-and-assumptions.md, rule 9.
"""
import argparse
import json
import sys

EXAMPLE = {
    "currency": "Mio. EUR",
    "opening_liquidity": 138.2,
    "horizon_start": 2026,
    "horizon_end": 2035,
    "schedule_years": [2026, 2027, 2028],
    "sensitivity_horizons": [2030, 2035, 2040],
    "revenue_base": 203.985,
    "earnings_base": 27.278,
    # The rate the IRR must beat, in percent, plus where it comes from. The
    # script refuses a hurdle without a stated origin - see hurdle().
    "hurdle": 11.1,
    "hurdle_source": "return on average equity 2025, annual report p. 49 / p. 51",
    "revenue_scenarios": [
        {"label": "base year held constant", "revenue": 203.985},
        {"label": "+3 % p.a.", "growth": 0.03, "years": 3},
        {"label": "prior-year level", "revenue": 312.346},
    ],
    "options": {
        "A": {
            "name": "Diversification into adjacent sectors",
            "investment": {"2026": 11.7, "2027": 11.7, "2028": 11.6},
            "working_capital": {"2027": 7.5, "2028": 7.5},
            "earnings_uplift": 16.0,
            "uplift_start": 2028,
            "remarks": {"2026": "sales and application build-up",
                        "2027": "market development; working capital build-up",
                        "2028": "first full earnings contribution"},
        },
        "B": {
            "name": "Adjacent production technology",
            "investment": {"2026": 10.0, "2027": 10.0, "2028": 10.0},
            "working_capital": {"2026": 7.5, "2027": 7.5},
            "earnings_uplift": 14.0,
            "uplift_start": 2027,
            "remarks": {"2026": "R&D focus", "2027": "market entry",
                        "2028": "scale-up"},
        },
        "C": {
            "name": "Acquisition",
            "investment": {"2026": 45.0, "2027": 7.5, "2028": 7.5},
            "working_capital": {"2026": 20.0},
            "earnings_uplift": 10.0,
            "uplift_start": 2027,
            "remarks": {"2026": "closing; purchase price and working capital taken on",
                        "2027": "integration; first contribution",
                        "2028": "synergies realised"},
        },
    },
}


# --- model ------------------------------------------------------------------

def _year_amount(mapping: dict, year: int) -> float:
    """Amount for a year in a {year: amount} mapping whose keys may be str or int."""
    return float(mapping.get(str(year), mapping.get(year, 0.0)))


def uplift(opt: dict, year: int) -> float:
    """Additional earnings of the option in the given year."""
    return float(opt["earnings_uplift"]) if year >= int(opt["uplift_start"]) else 0.0


def operating_cf(opt: dict, year: int) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Incremental operating cash flow: earnings uplift less the build-up of
        tied capital in that year.

    Inputs:
        opt (dict): one option from the config.
        year (int): calendar year.

    Outputs:
        cf (float): cash flow of the year, in the config's currency unit.
    --------------------------------------------------------------------------
    """
    return uplift(opt, year) - _year_amount(opt.get("working_capital", {}), year)


def schedule(cfg: dict, key: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Per-year table for one option over cfg["schedule_years"].

    Inputs:
        cfg (dict): full configuration.
        key (str): option key.

    Outputs:
        rows (list[dict]): year, investment, op_cf, liquidity, remark.
    --------------------------------------------------------------------------
    """
    opt = cfg["options"][key]
    liq = float(cfg["opening_liquidity"])
    rows = []
    end = int(cfg["horizon_end"])
    terminal = float(opt.get("terminal_value", 0.0))
    for year in cfg["schedule_years"]:
        inv = _year_amount(opt.get("investment", {}), year)
        ocf = operating_cf(opt, year)
        bemerkung = opt.get("remarks", {}).get(str(year),
                    opt.get("remarks", {}).get(year, ""))
        # Faellt das Horizontende in die gezeigten Jahre, muss der Endwert hier
        # sichtbar sein: eine Tabelle, die ihn verschweigt, widerspricht dem
        # IRR darunter, und der Leser kann die Differenz nicht aufloesen.
        if year == end and terminal:
            ocf += terminal
            hinweis = f"incl. terminal value {_n(terminal)}"
            bemerkung = f"{bemerkung}; {hinweis}" if bemerkung else hinweis
        liq = liq - inv + ocf
        rows.append({
            "year": year,
            "investment": inv,
            "op_cf": ocf,
            "liquidity": liq,
            "remark": bemerkung,
        })
    return rows


def full_cashflows(cfg: dict, key: str, horizon_end: int = None,
                   investment_scale: float = 1.0) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Payment series over the return horizon. The uplift continues at a
        constant level from its start year; tied capital is released in the
        final year of the horizon, together with the option's terminal value
        if one is configured.

    Inputs:
        cfg (dict): full configuration.
        key (str): option key.
        horizon_end (int): overrides cfg["horizon_end"].
        investment_scale (float): multiplies every investment outflow. 1.0 is
            the configured case; break_even_investment() varies it to find the
            price at which the return still clears the hurdle.

    Outputs:
        flows (list[float]): one value per year from horizon_start to horizon_end.
    --------------------------------------------------------------------------
    """
    opt = cfg["options"][key]
    end = int(horizon_end if horizon_end is not None else cfg["horizon_end"])
    wc_total = sum(float(v) for v in opt.get("working_capital", {}).values())
    terminal = float(opt.get("terminal_value", 0.0))
    flows = []
    for year in range(int(cfg["horizon_start"]), end + 1):
        flow = (uplift(opt, year)
                - _year_amount(opt.get("working_capital", {}), year)
                - _year_amount(opt.get("investment", {}), year)
                * investment_scale)
        if year == end:
            # Tied capital comes back, and so does the stake itself when the
            # option is an ownership position rather than a spend. The terminal
            # value is NOT multiplied by investment_scale: an exit value does
            # not grow because the entry price did, and letting it scale would
            # make break_even_investment() return a meaningless price - both
            # sides of the comparison would move together.
            flow += wc_total + terminal
        flows.append(flow)
    return flows


def npv(cashflows: list, rate: float) -> float:
    """Net present value; the first element sits at t = 0."""
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cashflows))


def irr(cashflows: list, lo: float = -0.95, hi: float = 5.0, tol: float = 1e-10) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Internal rate of return by bisection.

    Inputs:
        cashflows (list[float]): payment series from t = 0.
        lo, hi (float): search bracket.
        tol (float): interval width at which to stop.

    Outputs:
        rate (float): decimal rate; 0.12 means 12 %.
    --------------------------------------------------------------------------
    """
    f_lo, f_hi = npv(cashflows, lo), npv(cashflows, hi)
    if f_lo * f_hi > 0:
        raise ValueError("no sign change in the search bracket - IRR undefined")
    for _ in range(500):
        mid = (lo + hi) / 2.0
        f_mid = npv(cashflows, mid)
        if abs(f_mid) < 1e-12 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def hurdle(cfg: dict) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Returns the rate the IRR has to beat, as a decimal. Read from
        cfg["hurdle"] (percent) and refused unless cfg["hurdle_source"] names
        where the figure comes from.

        An IRR alone supports no verdict. The hurdle must be a figure the
        report has already sourced - the subject's own return on equity is the
        usual defensible choice. A WACC assembled from a guessed beta and an
        assumed market premium is exactly the plausible-looking invention that
        rule 5 of rigor-and-assumptions.md forbids, so this function will not
        accept a hurdle without a stated origin.

    Inputs:
        cfg (dict): full configuration.

    Outputs:
        rate (float): decimal rate, or None when no hurdle is configured.
    --------------------------------------------------------------------------
    """
    if "hurdle" not in cfg:
        return None
    if not str(cfg.get("hurdle_source", "")).strip():
        raise ValueError(
            "cfg['hurdle'] is set but cfg['hurdle_source'] is empty. Name the "
            "figure and its page, e.g. 'return on average equity 2025, 11.1 %, "
            "10-K p. 49 and p. 51'. An unsourced hurdle is an invented one.")
    return float(cfg["hurdle"]) / 100.0


def break_even_investment(cfg: dict, key: str, horizon_end: int = None):
    """
    --------------------------------------------------------------------------
    Purpose:
        Largest investment total at which the option still reaches the hurdle.

        This is the number a decision actually needs. "Is the return good?"
        invites a yes or a no; "up to what price does it stay good?" can be
        compared against an asking price, a share price or a valuation, and it
        turns the analysis into something a reader can act on. For a listed
        share the investment stream is a single outflow at t = 0, so the scale
        factor lands directly on the break-even entry price.

        Bisection on a multiplier of the investment stream: the IRR falls
        monotonically as the outflow grows, so the bracket is well behaved.

    Inputs:
        cfg (dict): full configuration; needs "hurdle" and "hurdle_source".
        key (str): option key.
        horizon_end (int): overrides cfg["horizon_end"].

    Outputs:
        amount (float): investment total at the hurdle, or None when even a
            vanishing investment fails to reach it (then the earnings side,
            not the price, is what fails).
    --------------------------------------------------------------------------
    """
    rate = hurdle(cfg)
    if rate is None:
        raise ValueError("no hurdle configured - see hurdle() for why one is required")
    total = sum(float(v) for v
                in cfg["options"][key].get("investment", {}).values())
    if total <= 0:
        return None

    def ueber_hurdle(scale: float) -> bool:
        """Does this outflow still clear the hurdle?

        Decided on the net present value AT the hurdle rate, not on the IRR.
        For a conventional series the two are equivalent, and the NPV is defined
        where the IRR is not. That matters at both ends of the bracket: at a
        vanishing outflow an ownership position keeps its distributions and its
        exit value, so the series loses its sign change and irr() raises. The
        earlier code read that exception as 'below any positive hurdle', which
        is the exact opposite of a series made purely of inflows, and returned
        None instead of a price.
        """
        return npv(full_cashflows(cfg, key, horizon_end, scale), rate) > 0

    lo, hi = 1e-6, 50.0
    if not ueber_hurdle(lo):
        # Even a vanishing investment misses the hurdle: then the earnings
        # side fails, not the price, and no break-even price exists.
        return None
    if ueber_hurdle(hi):
        # The hurdle still holds at fifty times the configured outflow; a
        # returned price would be an artefact of the bracket, not a result.
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if ueber_hurdle(mid):
            lo = mid
        else:
            hi = mid
    return total * (lo + hi) / 2.0


def required_terminal_value(cfg: dict, key: str, horizon_end: int = None) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        The exit value the CURRENT investment already presupposes: what the
        stake must be worth at the end of the horizon for the configured price
        to meet the hurdle.

        This is the counter-question to break_even_investment(), and it is the
        one worth asking whenever the break-even price lands far below the
        market price. "Up to what price is it worth buying" then has an
        uncomfortable answer, and a reader reasonably asks what the market is
        assuming instead. Expressed as a multiple of the configured terminal
        value - book equity, say - it says plainly how much of today's price
        rests on the exit rather than on the distributions.

    Inputs:
        cfg (dict): full configuration; needs "hurdle" and "hurdle_source".
        key (str): option key.
        horizon_end (int): overrides cfg["horizon_end"].

    Outputs:
        value (float): terminal value at the hurdle. Negative means the
            distributions alone already clear it and no exit value is needed.
    --------------------------------------------------------------------------
    """
    rate = hurdle(cfg)
    if rate is None:
        raise ValueError("no hurdle configured - see hurdle() for why one is required")
    end = int(horizon_end if horizon_end is not None else cfg["horizon_end"])
    perioden = end - int(cfg["horizon_start"])

    ohne = dict(cfg["options"][key])
    ohne.pop("terminal_value", None)
    cfg_ohne = dict(cfg)
    cfg_ohne["options"] = dict(cfg["options"])
    cfg_ohne["options"][key] = ohne

    barwert = npv(full_cashflows(cfg_ohne, key, end), rate)
    return -barwert * (1.0 + rate) ** perioden


def required_growth(cfg: dict, key: str, horizon_end: int = None,
                    lo: float = -0.5, hi: float = 2.0):
    """
    --------------------------------------------------------------------------
    Purpose:
        The annual growth rate of the earnings stream that the CURRENT price
        already presupposes: what the uplift must compound at for the
        configured investment to meet the hurdle.

        This is the second inversion, and the one that repairs the standing
        weakness of a constant-level model. Holding the cash flow flat is what
        makes the analysis honest - it forecasts nothing - and it is also what
        makes every growing company look expensive, because the whole case sits
        in the term the model dropped. Inverting the price into a growth rate
        gives that term back WITHOUT forecasting: the report states the rate the
        price contains, and the reader checks it against the company's own
        realized history. "Do not buy" becomes "this price contains 26.7 % a
        year for ten years - has this company ever delivered that?", which is a
        question the primary documents can answer.

        Pair it with the multi-year series required by report-template.md
        section 2: a required rate without a realized distribution to compare it
        against is only half the argument.

    Inputs:
        cfg (dict): full configuration; needs "hurdle" and "hurdle_source".
        key (str): option key.
        horizon_end (int): overrides cfg["horizon_end"].
        lo, hi (float): bracket for the bisection, as decimal rates.

    Outputs:
        rate (float): decimal growth rate. Negative or zero means the stream
            already clears the hurdle without growth. None when even the upper
            bracket fails, which says the price cannot be justified by growth of
            the earnings stream at all - report that rather than widening the
            bracket until a number appears.
    --------------------------------------------------------------------------
    """
    rate = hurdle(cfg)
    if rate is None:
        raise ValueError("no hurdle configured - see hurdle() for why one is required")
    opt = cfg["options"][key]
    start = int(cfg["horizon_start"])
    end = int(horizon_end if horizon_end is not None else cfg["horizon_end"])
    basis = float(opt.get("earnings_uplift", 0.0))
    terminal = float(opt.get("terminal_value", 0.0))
    if basis <= 0:
        return None

    def barwert(g: float) -> float:
        flows = [-sum(float(v) for v in opt.get("investment", {}).values())]
        for t in range(1, end - start + 1):
            flows.append(basis * (1.0 + g) ** t)
        if len(flows) > 1:
            flows[-1] += terminal
        return npv(flows, rate)

    if barwert(hi) < 0:
        # Selbst am oberen Rand traegt der Preis nicht: das Wachstum ist nicht
        # die fehlende Groesse, sondern der Preis.
        return None
    for _ in range(200):
        mitte = (lo + hi) / 2.0
        if barwert(mitte) < 0:
            lo = mitte
        else:
            hi = mitte
    return (lo + hi) / 2.0


def margin(cfg: dict, key: str, revenue: float) -> float:
    """End-year margin in percent for a given revenue denominator."""
    base = float(cfg["earnings_base"])
    return (base + float(cfg["options"][key]["earnings_uplift"])) / revenue * 100.0


def net_debt_to_earnings(cfg: dict, key: str) -> float:
    """Net debt / earnings at the end of the schedule. Net cash gives a negative value."""
    liq = schedule(cfg, key)[-1]["liquidity"]
    base = float(cfg["earnings_base"]) + float(cfg["options"][key]["earnings_uplift"])
    return -liq / base


def scenario_revenue(cfg: dict, sc: dict) -> float:
    """Resolve a revenue scenario to a single denominator."""
    if "revenue" in sc:
        return float(sc["revenue"])
    base = float(cfg["revenue_base"])
    return base * (1.0 + float(sc["growth"])) ** int(sc.get("years", 0))


# --- rendering --------------------------------------------------------------

def _n(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}"


def _row(cells: list, fmt: str) -> str:
    return ("| " + " | ".join(cells) + " |") if fmt == "markdown" \
        else (" & ".join(cells) + r" \\")


def render_schedule(cfg: dict, key: str, fmt: str) -> str:
    unit = cfg.get("currency", "")
    rows = [_row([str(r["year"]), _n(r["investment"]), _n(r["op_cf"]),
                  _n(r["liquidity"]), r["remark"]], fmt)
            for r in schedule(cfg, key)]
    if fmt == "latex":
        return "\n".join(rows) + "\n"
    head = _row(["Year", f"Investment ({unit})", f"Operating CF ({unit})",
                 f"Liquidity after investment ({unit})", "Remark"], fmt)
    sep = _row(["---"] * 5, fmt)
    return "\n".join([head, sep] + rows) + "\n"


def render_comparison(cfg: dict, fmt: str) -> str:
    keys = list(cfg["options"])
    unit = cfg.get("currency", "")
    end_year = cfg["schedule_years"][-1]
    rev = float(cfg["revenue_base"])
    lines = [
        [f"Cumulative investment ({unit})"] +
        [_n(sum(float(v) for v in cfg["options"][k].get("investment", {}).values()))
         for k in keys],
        ["Earnings uplift from year"] +
        [f"{_n(float(cfg['options'][k]['earnings_uplift']))} from "
         f"{cfg['options'][k]['uplift_start']}" for k in keys],
        [f"Margin end {end_year} (%)"] + [_n(margin(cfg, k, rev)) for k in keys],
        [f"IRR (~), horizon {cfg['horizon_start']}-{cfg['horizon_end']} (%)"] +
        [_n(irr(full_cashflows(cfg, k)) * 100.0) for k in keys],
        [f"Liquidity reserve end {end_year} ({unit})"] +
        [_n(schedule(cfg, k)[-1]["liquidity"]) for k in keys],
        ["Net debt / earnings"] +
        [_n(net_debt_to_earnings(cfg, k), 2) for k in keys],
    ]
    rows = [_row(c, fmt) for c in lines]
    if fmt == "latex":
        return "\n".join(rows) + "\n"
    head = _row(["Metric"] + [f"{k} - {cfg['options'][k]['name']}" for k in keys], fmt)
    sep = _row(["---"] * (len(keys) + 1), fmt)
    return "\n".join([head, sep] + rows) + "\n"


def render_irr_sensitivity(cfg: dict, fmt: str) -> str:
    keys = list(cfg["options"])
    start = int(cfg["horizon_start"])
    rows = []
    for end in cfg["sensitivity_horizons"]:
        cells = [f"{int(end) - start + 1} years (to {end})"] + \
                [_n(irr(full_cashflows(cfg, k, horizon_end=int(end))) * 100.0) for k in keys]
        rows.append(_row(cells, fmt))
    if fmt == "latex":
        return "\n".join(rows) + "\n"
    head = _row(["Horizon"] + [f"{k} (%)" for k in keys], fmt)
    sep = _row(["---"] * (len(keys) + 1), fmt)
    return "\n".join([head, sep] + rows) + "\n"


def render_margin_sensitivity(cfg: dict, fmt: str) -> str:
    keys = list(cfg["options"])
    rows = []
    for sc in cfg.get("revenue_scenarios", []):
        rev = scenario_revenue(cfg, sc)
        cells = [f"{sc['label']} ({_n(rev)})"] + [_n(margin(cfg, k, rev)) for k in keys]
        rows.append(_row(cells, fmt))
    if not rows:
        return ""
    if fmt == "latex":
        return "\n".join(rows) + "\n"
    head = _row(["Revenue base"] + [f"{k} (%)" for k in keys], fmt)
    sep = _row(["---"] * (len(keys) + 1), fmt)
    return "\n".join([head, sep] + rows) + "\n"


# --- cli --------------------------------------------------------------------

def render_break_even(cfg: dict, fmt: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Table of the break-even investment per option and horizon, against the
        sourced hurdle. Returns "" when no hurdle is configured, so a report
        that cannot source one simply omits the table rather than inventing it.

    Inputs:
        cfg (dict): full configuration.
        fmt (str): "markdown" or "latex".

    Outputs:
        table (str): rendered table body, or "".
    --------------------------------------------------------------------------
    """
    if hurdle(cfg) is None:
        return ""
    keys = list(cfg["options"])
    horizons = cfg.get("sensitivity_horizons", [cfg["horizon_end"]])
    lines = []
    for h in horizons:
        cells = [f"to {h}"]
        for k in keys:
            amount = break_even_investment(cfg, k, h)
            cells.append("below hurdle" if amount is None else _n(amount))
        lines.append(_row(cells, fmt))
    return "\n".join(lines) + "\n"


def build_outputs(cfg: dict, fmt: str) -> dict:
    out = {f"cf_{k.lower()}": render_schedule(cfg, k, fmt) for k in cfg["options"]}
    out["comparison"] = render_comparison(cfg, fmt)
    out["irr_sensitivity"] = render_irr_sensitivity(cfg, fmt)
    ms = render_margin_sensitivity(cfg, fmt)
    if ms:
        out["margin_sensitivity"] = ms
    be = render_break_even(cfg, fmt)
    if be:
        out["break_even"] = be
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", help="path to the JSON configuration")
    ap.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    ap.add_argument("--outdir", help="write one file per table instead of stdout")
    ap.add_argument("--example", action="store_true",
                    help="print a complete example configuration and exit")
    args = ap.parse_args(argv[1:])

    if args.example:
        print(json.dumps(EXAMPLE, indent=2))
        return 0
    if not args.config:
        ap.error("--config is required (or use --example)")

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)

    for field in ("opening_liquidity", "horizon_start", "horizon_end",
                  "schedule_years", "options", "revenue_base", "earnings_base"):
        if field not in cfg:
            print(f"config is missing required field: {field}", file=sys.stderr)
            return 2
    if not cfg["options"]:
        print("config defines no options", file=sys.stderr)
        return 2

    outputs = build_outputs(cfg, args.format)
    ext = "tex" if args.format == "latex" else "md"

    if args.outdir:
        import os
        os.makedirs(args.outdir, exist_ok=True)
        header = "% generated by cashflow_irr.py - do not edit by hand\n" \
            if args.format == "latex" else \
            "<!-- generated by cashflow_irr.py - do not edit by hand -->\n"
        for name, body in outputs.items():
            path = os.path.join(args.outdir, f"{name}.{ext}")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(header + body)
            print(f"OK -> {path}")
    else:
        for name, body in outputs.items():
            print(f"\n### {name}\n")
            print(body, end="")

    print("\nReminder: publish the IRR horizon "
          f"({cfg['horizon_start']}-{cfg['horizon_end']}) and the assumptions "
          "behind the investment profile and working-capital timing.", file=sys.stderr)
    if "hurdle" not in cfg:
        print("No hurdle configured: the IRRs above compare the options with "
              "each other but support no verdict. Add \"hurdle\" and "
              "\"hurdle_source\" (a figure the report already sourced, e.g. "
              "return on average equity) to get the break-even table.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
