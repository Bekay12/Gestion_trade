#!/usr/bin/env python3
"""
run_scheduled - Dispatcher du cycle gaps quotidien, cron local (option A, decidee le 09.10.2026
apres le blocage reseau des routines cloud : finviz.com, data.sec.gov et
academygermain.com n'etaient pas dans la liste blanche du bac a sable cloud ;
cette machine n'a pas cette restriction).

Appele toutes les 5 minutes par cron. Ne declenche une tache que si l'heure
reelle de New York (pas l'heure systeme de cette machine, qui est en Europe)
tombe dans la fenetre de son creneau ET si la tache n'a pas deja ete faite
aujourd'hui (fichier marqueur dans .done/). Ce design survit aux changements
d'heure d'ete US et UE, qui ne tombent pas le meme jour (ete UE : 25.10.2026 ;
ete US : 01.11.2026) - un cron a horaire fixe en heure locale europeenne se
serait desynchronise pendant cette semaine d'ecart, et deux fois par an
ensuite.

Creneaux (heure de New York, jours ouvres) :
    09:10  premarket   - gap_scan.py --mode premarket (repousse du 08:00 au
                         08:45 puis au 09:10 le 14.10.2026, sur demande
                         explicite, pour laisser le gap pre-marche se former
                         le plus longtemps possible avant le scan tout en
                         restant dans la fenetre premarket de gap_scan.py
                         (7h00-9h30 ET, cf. _fenetre_actuelle). Motif : le
                         14.10, 5 titres (FFR/RKDA/AGMH/WFF/CDT) montraient un
                         gap haussier a 08h45 ET et avaient deja bascule en
                         gap BAISSIER a l'ouverture officielle 9h30 ET - un
                         scan plus tardif reduit, sans l'annuler, la fenetre
                         ou un pump peut se degonfler sans etre vu.
    13:00  cassure_midi - cassure_scan.py (suffixe HHMM)
    16:05  cloture     - cassure_scan.py + gap_scan.py --mode close
    17:00  bilan       - evaluer_gaps.py + germain_revues.py + bilan factuel

Portee identique a celle deja actee pour les routines cloud (Phase 1) :
aucune lecture de depot SEC, aucun jugement de catalyseur, aucune
evaluation. Purement mecanique ; le jugement reste manuel.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

GAPS = Path(__file__).resolve().parent.parent
TRADING_AGENT = GAPS.parent
SKILL = TRADING_AGENT / "skills" / "gap-trading-germain"
DONE_DIR = Path(__file__).resolve().parent / ".done"
VENV_PYTHON = TRADING_AGENT.parent / ".venv_new" / "bin" / "python3"

sys.path.insert(0, str(GAPS))
from notify import notify  # noqa: E402

NY = ZoneInfo("America/New_York")

# (heure, minute, tolerance_minutes, nom_du_creneau)
SLOTS = [
    (9, 10, 4, "premarket"),
    (13, 0, 4, "cassure_midi"),
    (16, 5, 4, "cloture"),
    (17, 0, 4, "bilan"),
]


def _ny_now() -> datetime:
    return datetime.now(NY)


def _done_marker(slot: str, day: date) -> Path:
    return DONE_DIR / f"{day.isoformat()}_{slot}.done"


def _already_done(slot: str, day: date) -> bool:
    return _done_marker(slot, day).exists()


def _mark_done(slot: str, day: date) -> None:
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    _done_marker(slot, day).touch()


def _run(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def _grouper_par_classe(verdicts: list[dict]) -> str:
    """Tickers detectes groupes par classe, pour que la notification se lise sans ouvrir
    le fichier JSON. Decision du 12.10.2026 (demande explicite : moins d'allers-retours).

    Chaque groupe porte sa direction ([VENDEUSE]/[ACHETEUSE], champ "direction"
    deja serialise par gap_qualifier.rendre_verdict - FADE/PUMP_RISK vendeuse,
    CONTINUATION/SQUEEZE acheteuse, A_CONFIRMER/INSUFFISANT aucune) - demande
    du 14.10.2026, pour voir d'un coup d'oeil si la liste du jour est a sens
    unique ou melangee, sans devoir se souvenir de la classification par classe."""
    par_classe: dict[str, list[str]] = {}
    direction_par_classe: dict[str, str | None] = {}
    for v in verdicts:
        classe = v.get("classe", "?")
        par_classe.setdefault(classe, []).append(v.get("ticker", "?"))
        direction_par_classe.setdefault(classe, v.get("direction"))
    if not par_classe:
        return "aucun titre detecte"
    def _etiquette(classe: str) -> str:
        d = direction_par_classe.get(classe)
        return f" [{d.upper()}]" if d else ""
    return " | ".join(f"{classe}{_etiquette(classe)} ({len(tickers)}): {', '.join(tickers)}"
                      for classe, tickers in sorted(par_classe.items()))


def _resume_bilan(notes: list[dict], alertes_par_ticker: dict[str, list[str]] | None = None) -> str:
    """Detail du bilan (justesse par classe, alertes trop severes, shorts fondes,
    mouvements notables ecartes a la detection) pour la notification du soir.

    Pour PUMP_RISK, la ligne par classe affiche toujours la scission "avec depot
    SEC / sans" (piste ouverte le 13.10.2026 : un verdict appuye sur un depot reel
    se verifie plus souvent qu'un verdict purement comportemental - voir motifs
    contenant "depot"). Affichee meme quand un sous-groupe est vide (0/0) - demande
    du 16.10.2026, pour voir d'un jour a l'autre si le depot reste absent plutot
    que de devoir deviner pourquoi la ligne a disparu. Observationnel pour
    l'instant, pas une regle de decision ; suivi ici pour voir si ca se confirme
    sur plus de jours."""
    alertes_par_ticker = alertes_par_ticker or {}
    juges = [n for n in notes if n.get("resultat") in ("juste", "faux")]
    n_juges = len(juges)
    n_justes = sum(1 for n in juges if n["resultat"] == "juste")
    taux = round(100 * n_justes / n_juges) if n_juges else 0
    lignes = [f"{n_justes}/{n_juges} justes ({taux}%)"]

    par_classe: dict[str, list[dict]] = {}
    for n in juges:
        par_classe.setdefault(n.get("classe", "?"), []).append(n)
    for classe, ns in sorted(par_classe.items()):
        j = sum(1 for n in ns if n["resultat"] == "juste")
        if classe == "PUMP_RISK":
            avec = [n for n in ns if any("depot" in a for a in
                                         alertes_par_ticker.get(n["ticker"], []))]
            sans = [n for n in ns if n not in avec]
            ja = sum(1 for n in avec if n["resultat"] == "juste")
            js = sum(1 for n in sans if n["resultat"] == "juste")
            lignes.append(f"{classe}: {j}/{len(ns)} juste "
                          f"(depot SEC {ja}/{len(avec)}, sans {js}/{len(sans)})")
            continue
        lignes.append(f"{classe}: {j}/{len(ns)} juste")

    severes = [n["ticker"] for n in juges
              if n.get("classe") == "PUMP_RISK" and n["resultat"] == "faux"]
    if severes:
        lignes.append(f"PUMP_RISK trop severe: {', '.join(severes)}")

    fondes = [n["ticker"] for n in juges
             if n.get("classe") == "PUMP_RISK" and n["resultat"] == "juste"
             and n.get("executabilite", {}).get("executable") == "oui"]
    if fondes:
        lignes.append(f"Shorts fondes+executables: {', '.join(fondes)}")

    non_juges = [n for n in notes if n.get("resultat") not in ("juste", "faux")]
    # variation_pct peut valoir None (non mesurable), pas seulement absent -
    # .get(cle, 0) ne couvre pas ce cas, abs(None) levait une TypeError
    # avalee par le except Exception de task_bilan (bug du 13.10.2026, note
    # VRXA "non juge" a variation_pct=None).
    notables = [f"{n['ticker']} ({(n.get('variation_pct') or 0):+.1f}%)"
               for n in non_juges if abs(n.get("variation_pct") or 0) > 15]
    if notables:
        lignes.append(f"INSUFFISANT a mouvement notable: {', '.join(notables)}")

    return "\n".join(lignes)


def _git(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *cmd], cwd=str(TRADING_AGENT.parent),
                          capture_output=True, text=True)


def _commit_marker(day: date) -> Path:
    return DONE_DIR / f"{day.isoformat()}_committed.done"


def _already_committed_today(day: date) -> bool:
    return _commit_marker(day).exists()


def _stage(paths: list[str]) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Met en index les fichiers produits par une tache, SANS committer.
        Decision du 09.10.2026 : au plus un commit par jour, et seulement en
        cas d'echec de script ou le vendredi (bilan hebdomadaire) - une
        journee normale accumule juste des fichiers en index.

    Inputs:
        paths (list[str]): chemins relatifs a mettre en index

    Outputs:
        resume (str): compte-rendu court pour la notification
    --------------------------------------------------------------------------
    """
    add = _git(["add", *paths])
    if add.returncode != 0:
        return f"git add a echoue: {add.stderr[:200]}"
    return "mis en index (commit differe)"


def _commit_and_push(message: str, extra_paths: list[str] | None = None, day: date | None = None) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Le SEUL point du dispatcher qui commit reellement. Appele dans deux
        cas uniquement (decision du 09.10.2026) : un echec de script le jour
        meme, ou le bilan du vendredi qui cloture la semaine. Garde-fou "au
        plus un commit par jour" via un marqueur .done - si un commit
        d'erreur a deja eu lieu aujourd'hui, le commit hebdomadaire du
        vendredi est reporte a la semaine suivante plutot que d'en faire un
        second le meme jour.

    Inputs:
        message (str): message de commit
        extra_paths (list[str] | None): chemins supplementaires a mettre en
            index avant de committer (en plus de tout ce qui est deja en
            index depuis le debut de la semaine)
        day (date | None): jour courant, pour le marqueur anti-double-commit

    Outputs:
        resume (str): compte-rendu court pour la notification
    --------------------------------------------------------------------------
    """
    day = day or _ny_now().date()
    if _already_committed_today(day):
        return "commit du jour deja fait - report (max un commit/jour)"
    if extra_paths:
        _git(["add", *extra_paths])
    # git diff --cached, pas status --porcelain : ce depot a d'autres travaux
    # en cours non lies a gaps/ (voir la session du 09.10.2026) ; seul l'INDEX
    # (ce que _stage a ajoute) doit decider s'il y a quelque chose a committer,
    # jamais l'etat complet de l'arbre de travail.
    staged = _git(["diff", "--cached", "--quiet"])
    if staged.returncode == 0:  # 0 = pas de difference indexee
        return "rien a committer (index vide)"
    commit = _git(["commit", "-m", message])
    if commit.returncode != 0:
        return f"git commit a echoue: {commit.stderr[:200]}"
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    _commit_marker(day).touch()
    push = _git(["push"])
    if push.returncode != 0:
        return f"commit OK, push a echoue (a rejouer manuellement): {push.stderr[:200]}"
    return "committe et pousse"


def _verdicts_de(fichier: Path) -> list[dict]:
    try:
        return json.load(open(fichier)).get("verdicts", [])
    except Exception:
        return []


def _depots_sec_de(fichier: Path) -> dict[str, dict]:
    try:
        return json.load(open(fichier)).get("depots_sec", {})
    except Exception:
        return {}


def _resume_depots_sec(depots_sec: dict[str, dict]) -> str:
    """Tickers avec un depot SEC recent, pour que la notification premarket le
    signale sans ouvrir le fichier JSON - demande du 14.10.2026. Deja calcule
    par edgar_depots.py et ecrit dans le JSON par gap_scan.py ; aucun appel
    reseau supplementaire ici.

    Deux groupes, parce qu'ils ne pesent pas pareil :
        dilutif    : formulaire_sec classe (424B3/424B5/S-1) - catalyseur
                     baissier deguise, meme logique que _resume_bilan.
        a verifier : depot recent (ex. 6-K) non encore classe comme dilutif -
                     catalyseur possible, a lire avant d'agir."""
    dilutifs = {t: d["formulaire_sec"] for t, d in depots_sec.items() if d.get("formulaire_sec")}
    a_verifier = {t: sorted({dep.get("form", "?") for dep in d.get("a_lire", [])})
                 for t, d in depots_sec.items()
                 if d.get("a_lire") and not d.get("formulaire_sec")}
    if not dilutifs and not a_verifier:
        return "Depot SEC : aucun"
    parts = []
    if dilutifs:
        parts.append("dilutif " + ", ".join(f"{t} ({f})" for t, f in sorted(dilutifs.items())))
    if a_verifier:
        parts.append("a verifier " + ", ".join(f"{t} ({'/'.join(fs)})"
                                               for t, fs in sorted(a_verifier.items())))
    return "Depot SEC : " + " | ".join(parts)


def task_premarket(today: date) -> bool:
    out = GAPS / "detections" / f"{today.isoformat()}_premarket.json"
    r = _run([str(VENV_PYTHON), "scripts/gap_scan.py", "--mode", "premarket",
             "--min-gap", "5", "--limit", "30", "--json", str(out)], cwd=SKILL)
    resume_git = _stage([str(out.relative_to(TRADING_AGENT.parent))])
    ok = r.returncode == 0
    detail = (f"{_grouper_par_classe(_verdicts_de(out))}\n"
             f"{_resume_depots_sec(_depots_sec_de(out))}") if ok else ""
    notify("Gaps - premarket",
          f"{detail}\n{resume_git}" if ok
          else f"echec (code {r.returncode}): {r.stderr[-300:]}")
    return ok


def task_cassure_midi(today: date) -> bool:
    hhmm = _ny_now().strftime("%H%M")
    out = GAPS / "detections" / f"{today.isoformat()}_cassure_{hhmm}.json"
    r = _run([str(VENV_PYTHON), "scripts/cassure_scan.py", "--min-hausse", "5",
             "--limit", "50", "--json", str(out)], cwd=SKILL)
    resume_git = _stage([str(out.relative_to(TRADING_AGENT.parent))])
    ok = r.returncode == 0
    detail = _grouper_par_classe(_verdicts_de(out)) if ok else ""
    notify("Gaps - cassure midi",
          f"{detail}\n{resume_git}" if ok
          else f"echec (code {r.returncode}): {r.stderr[-300:]}")
    return ok


def task_cloture(today: date) -> bool:
    hhmm = _ny_now().strftime("%H%M")
    out_cassure = GAPS / "detections" / f"{today.isoformat()}_cassure_{hhmm}.json"
    out_close = GAPS / "detections" / f"{today.isoformat()}_close.json"
    r1 = _run([str(VENV_PYTHON), "scripts/cassure_scan.py", "--min-hausse", "5",
              "--limit", "50", "--json", str(out_cassure)], cwd=SKILL)
    r2 = _run([str(VENV_PYTHON), "scripts/gap_scan.py", "--mode", "close",
              "--min-gap", "5", "--json", str(out_close)], cwd=SKILL)
    resume_git = _stage([str(out_cassure.relative_to(TRADING_AGENT.parent)),
                        str(out_close.relative_to(TRADING_AGENT.parent))])
    ok = r1.returncode == 0 and r2.returncode == 0
    detail = (f"Cassures: {_grouper_par_classe(_verdicts_de(out_cassure))}\n"
             f"Close: {_grouper_par_classe(_verdicts_de(out_close))}") if ok else ""
    notify("Gaps - cloture",
          f"{detail}\n{resume_git}" if ok
          else f"echec partiel (cassure={r1.returncode}, close={r2.returncode})")
    return ok


def task_bilan(today: date) -> bool:
    day = today.isoformat()
    close_file = GAPS / "detections" / f"{day}_close.json"
    premarket_file = GAPS / "detections" / f"{day}_premarket.json"
    chosen = close_file if close_file.exists() else (
        premarket_file if premarket_file.exists() else None)
    catalyseurs = GAPS / "catalyseurs" / f"{day}.md"
    if chosen is None:
        catalyseurs.write_text(
            f"# Bilan gaps : {day}\n\nAucune detection premarket/close disponible "
            "pour cette date au moment du bilan automatique.\n")
        _stage([str(catalyseurs.relative_to(TRADING_AGENT.parent))])
        notify("Gaps - bilan du soir", "aucune detection disponible pour aujourd'hui")
        _maybe_weekly_commit(today)
        return True

    r_eval = _run([str(VENV_PYTHON), "evaluer_gaps.py", "--fichier",
                  str(chosen.relative_to(GAPS)), "--jours", "1"], cwd=GAPS)
    r_germain = _run([str(VENV_PYTHON), "scripts/germain_revues.py", "--depuis", day,
                      "--sortie", str((GAPS / "germain").relative_to(SKILL.parent.parent))],
                     cwd=SKILL)

    germain_files = sorted((GAPS / "germain").glob(f"{day}_*.json"))
    germain_titles = []
    for f in germain_files:
        try:
            d = json.load(open(f))
            germain_titles.append((d.get("titre", f.name),
                                   [l.get("ticker") for l in d.get("lignes", [])]))
        except Exception:
            continue

    lines = [f"# Bilan gaps : {day}", "", "## Notation du soir", "", "```",
             r_eval.stdout.strip() or "(aucune sortie)", "```", "",
             "## Revues Academy Germain du jour", ""]
    if germain_titles:
        for titre, tickers in germain_titles:
            lines.append(f"- {titre} — {', '.join(tickers) if tickers else '(aucun ticker)'}")
    else:
        lines.append("Aucune revue Academy Germain publiee pour cette date au moment du bilan.")

    our_tickers = set()
    alertes_par_ticker: dict[str, list[str]] = {}
    try:
        detection_data = json.load(open(chosen))
        our_tickers = {v.get("ticker") for v in detection_data.get("verdicts", [])}
        alertes_par_ticker = {v.get("ticker"): v.get("alertes", [])
                              for v in detection_data.get("verdicts", [])}
    except Exception:
        pass
    germain_tickers = {t for _, tl in germain_titles for t in tl if t}
    lines += ["", "## Recoupement", "",
             f"- Communs : {sorted(our_tickers & germain_tickers) or '(aucun)'}",
             f"- Uniquement dans nos detections : {sorted(our_tickers - germain_tickers) or '(aucun)'}",
             f"- Uniquement chez Germain : {sorted(germain_tickers - our_tickers) or '(aucun)'}"]

    catalyseurs.write_text("\n".join(lines) + "\n")
    resume_stage = _stage([str(catalyseurs.relative_to(TRADING_AGENT.parent)),
                          str((GAPS / "germain").relative_to(TRADING_AGENT.parent))])
    ok = r_eval.returncode == 0  # germain_revues peut legitimement rendre 0 revue, pas un echec

    eval_file = chosen.with_name(chosen.stem + "_evalue_J1.json")
    detail_bilan = "(detail non disponible)"
    if eval_file.exists():
        try:
            detail_bilan = _resume_bilan(json.load(open(eval_file)).get("notes", []),
                                         alertes_par_ticker)
        except Exception:
            pass
    germain_info = (f"Germain : {len(germain_titles)} revue(s) publiee(s)" if germain_titles
                   else "Germain : aucune revue publiee au moment du bilan")
    notify("Gaps - bilan du soir",
          f"Bilan {day}\n{detail_bilan}\n{germain_info}" if ok
          else f"evaluer_gaps a echoue (code {r_eval.returncode})")
    weekly = _maybe_weekly_commit(today)
    if weekly:
        notify("Gaps - commit hebdomadaire", weekly)
    return ok


def _maybe_weekly_commit(today: date) -> str | None:
    """Vendredi uniquement (decision du 09.10.2026) : committe tout ce qui est
    en index depuis le debut de la semaine, en un seul commit. Rien ne se
    passe les autres jours - c'est task_bilan qui appelle ceci en dernier."""
    if today.weekday() != 4:  # 4 = vendredi
        return None
    return _commit_and_push(f"bilan hebdomadaire, semaine du {today.isoformat()} (cron local)",
                            day=today)


TASKS = {"premarket": task_premarket, "cassure_midi": task_cassure_midi,
        "cloture": task_cloture, "bilan": task_bilan}


def main() -> int:
    now = _ny_now()
    if now.weekday() >= 5:  # samedi, dimanche
        return 0
    today = now.date()
    for hour, minute, tolerance, slot in SLOTS:
        target_minutes = hour * 60 + minute
        now_minutes = now.hour * 60 + now.minute
        if abs(now_minutes - target_minutes) > tolerance:
            continue
        if _already_done(slot, today):
            continue
        try:
            ok = TASKS[slot](today)
        except Exception as e:
            notify(f"Gaps - {slot} - ERREUR", f"{type(e).__name__}: {e}", priority="high")
            resume = _commit_and_push(
                f"ERREUR {slot} {today.isoformat()} (cron local) : {type(e).__name__}",
                day=today)
            notify(f"Gaps - {slot} - commit d'erreur", resume, priority="high")
        else:
            _mark_done(slot, today)
            if not ok:
                resume = _commit_and_push(
                    f"ECHEC {slot} {today.isoformat()} (cron local)", day=today)
                notify(f"Gaps - {slot} - commit d'erreur", resume, priority="high")
    return 0


if __name__ == "__main__":
    sys.exit(main())
