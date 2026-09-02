"""
Verrouillage du garde-fou de chargement du module C.

Le .so du 2026-04-18 avait ete compile avec AddressSanitizer. Son chargement ne
leve pas une exception : il fait AVORTER le processus (« ASan runtime does not
come first in initial library list »), donc le try/except de _diagnose_import
est impuissant. Consequence mesuree : `python optimisateur_hybride.py` mourait a
l'import, sans message Python. Le refus doit donc avoir lieu AVANT le dlopen.
"""
from trading_c_acceleration import qsi_optimized


def test_binaire_instrumente_detecte(tmp_path) -> None:
    """Un binaire portant les symboles ASan est reconnu."""
    faux = tmp_path / "trading_c.cpython-310-x86_64-linux-gnu.so"
    faux.write_bytes(b"\x7fELF" + b"\x00" * 64 + b"__asan_init" + b"\x00" * 16)

    assert qsi_optimized._so_instrumente(str(faux)) is True


def test_binaire_propre_accepte(tmp_path) -> None:
    """Un binaire sans symbole ASan passe."""
    propre = tmp_path / "trading_c.cpython-310-x86_64-linux-gnu.so"
    propre.write_bytes(b"\x7fELF" + b"\x00" * 512)

    assert qsi_optimized._so_instrumente(str(propre)) is False


def test_fichier_absent_ne_leve_pas(tmp_path) -> None:
    """Un chemin invalide ne doit pas casser le chargement."""
    assert qsi_optimized._so_instrumente(str(tmp_path / "absent.so")) is False


def test_candidats_filtres_par_nom_et_extension(monkeypatch, tmp_path) -> None:
    """Seules les bibliotheques compilees du bon module sont candidates."""
    (tmp_path / "trading_c.cpython-310-x86_64-linux-gnu.so").write_bytes(b"x")
    (tmp_path / "trading_c.pyd").write_bytes(b"x")
    (tmp_path / "autre_module.so").write_bytes(b"x")
    (tmp_path / "trading_c.py").write_text("# pas un binaire")
    monkeypatch.setattr(qsi_optimized, "__file__", str(tmp_path / "qsi_optimized.py"))

    trouves = {p.rsplit("/", 1)[-1] for p in qsi_optimized._binaires_candidats("trading_c")}

    assert trouves == {"trading_c.cpython-310-x86_64-linux-gnu.so", "trading_c.pyd"}
