# Fenêtre dédiée à l'optimisation hybride
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QMessageBox, QTabWidget, QGroupBox
)
from PyQt5.QtCore import Qt

sys.path.append(str(Path(__file__).parent.parent))

try:
    from symbol_manager import get_all_sectors, get_all_cap_ranges, get_symbols_by_sector_and_cap
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    SYMBOL_MANAGER_AVAILABLE = False

from optimisateur_hybride import clean_sector_cap_groups


class OptimizationWindow(QMainWindow):
    """Fenêtre dédiée à l'optimisation hybride avec paramètres modifiables."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimiseur Hybride - Paramètres & Résultats")
        self.setGeometry(100, 100, 1400, 900)
        
        if not SYMBOL_MANAGER_AVAILABLE:
            QMessageBox.critical(self, "Erreur", "SQLite/symbol_manager non disponible")
            return
        
        self.init_ui()
    
    def init_ui(self):
        """Initialiser l'interface utilisateur."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        
        # ========== PANNEAU GAUCHE : Paramètres ==========
        left_panel = self.create_parameters_panel()
        
        # ========== PANNEAU DROIT : Résultats ==========
        right_panel = self.create_results_panel()
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        main_widget.setLayout(main_layout)
    
    def create_parameters_panel(self):
        """Créer le panneau des paramètres."""
        group = QGroupBox("⚙️ Paramètres d'optimisation")
        layout = QVBoxLayout()
        
        # ---- Min/Max symboles ----
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Min symboles par groupe:"), 0, 0)
        self.min_symbols_spin = QSpinBox()
        self.min_symbols_spin.setMinimum(1)
        self.min_symbols_spin.setMaximum(100)
        self.min_symbols_spin.setValue(6)
        grid.addWidget(self.min_symbols_spin, 0, 1)
        
        grid.addWidget(QLabel("Max symboles par groupe:"), 1, 0)
        self.max_symbols_spin = QSpinBox()
        self.max_symbols_spin.setMinimum(1)
        self.max_symbols_spin.setMaximum(500)
        self.max_symbols_spin.setValue(15)
        grid.addWidget(self.max_symbols_spin, 1, 1)
        
        # ---- Ratio fixe/aléatoire ----
        grid.addWidget(QLabel("Ratio FIXE (mes_symbols):"), 2, 0)
        self.fixed_ratio_spin = QDoubleSpinBox()
        self.fixed_ratio_spin.setMinimum(0.0)
        self.fixed_ratio_spin.setMaximum(1.0)
        self.fixed_ratio_spin.setSingleStep(0.05)
        self.fixed_ratio_spin.setValue(0.6)
        self.fixed_ratio_spin.setDecimals(2)
        self.fixed_ratio_spin.setSuffix(" (60% = défaut)")
        grid.addWidget(self.fixed_ratio_spin, 2, 1)
        
        # ---- Cache TTL ----
        grid.addWidget(QLabel("Cache TTL (jours):"), 3, 0)
        self.ttl_days_spin = QSpinBox()
        self.ttl_days_spin.setMinimum(0)
        self.ttl_days_spin.setMaximum(365)
        self.ttl_days_spin.setValue(0)
        self.ttl_days_spin.setSuffix(" (0 = désactivé)")
        grid.addWidget(self.ttl_days_spin, 3, 1)
        
        layout.addLayout(grid)
        layout.addSpacing(20)
        
        # ---- Symboles d'optimisation ----
        layout.addWidget(QLabel("📋 Symboles d'optimisation:"))
        self.optim_list = QListWidget()
        self.optim_list.setMaximumHeight(250)
        layout.addWidget(self.optim_list)
        
        # Compteur
        self.optim_count_label = QLabel("0 symboles")
        layout.addWidget(self.optim_count_label)
        
        layout.addSpacing(20)
        
        # ---- Boutons d'action ----
        button_layout = QVBoxLayout()
        
        self.run_button = QPushButton("▶️ Lancer l'optimisation")
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_button.clicked.connect(self.run_optimization)
        button_layout.addWidget(self.run_button)
        
        self.reload_button = QPushButton("🔄 Recharger symboles")
        self.reload_button.clicked.connect(self.reload_optimization_symbols)
        button_layout.addWidget(self.reload_button)
        
        self.clear_cache_button = QPushButton("🗑️ Vider cache")
        self.clear_cache_button.clicked.connect(self.clear_cache)
        button_layout.addWidget(self.clear_cache_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_results_panel(self):
        """Créer le panneau des résultats."""
        group = QGroupBox("📊 Résultats & Logs")
        layout = QVBoxLayout()
        
        # Tabs : Résumé / Détails / Logs
        tabs = QTabWidget()
        
        # ---- Tab 1 : Résumé ----
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        tabs.addTab(self.summary_text, "📈 Résumé")
        
        # ---- Tab 2 : Détails groupes ----
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        tabs.addTab(self.details_text, "🔍 Détails")
        
        # ---- Tab 3 : Logs ----
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        tabs.addTab(self.logs_text, "📝 Logs")
        
        layout.addWidget(tabs)
        
        # ---- Boutons de gestion ----
        manage_layout = QHBoxLayout()
        
        self.export_button = QPushButton("💾 Exporter résultats")
        self.export_button.clicked.connect(self.export_results)
        manage_layout.addWidget(self.export_button)
        
        self.copy_button = QPushButton("📋 Copier liste")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        manage_layout.addWidget(self.copy_button)
        
        layout.addLayout(manage_layout)
        
        group.setLayout(layout)
        return group
    
    def reload_optimization_symbols(self):
        """Recharger la liste des symboles d'optimisation depuis SQLite."""
        try:
            from symbol_manager import get_symbols_by_list_type
            symbols = get_symbols_by_list_type(list_type="optimization", active_only=True)
            
            self.optim_list.clear()
            for sym in symbols:
                item = QListWidgetItem(sym)
                item.setData(Qt.UserRole, sym)
                self.optim_list.addItem(item)
            
            self.optim_count_label.setText(f"{len(symbols)} symboles")
            self.logs_text.append(f"✅ {len(symbols)} symboles chargés depuis SQLite")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les symboles: {e}")
            self.logs_text.append(f"❌ Erreur: {e}")
    
    def run_optimization(self):
        """Lancer l'optimisation hybride."""
        try:
            self.logs_text.clear()
            self.summary_text.clear()
            self.details_text.clear()
            
            self.logs_text.append("🚀 Démarrage de l'optimisation hybride...\n")
            
            # Récupérer les paramètres
            min_sym = self.min_symbols_spin.value()
            max_sym = self.max_symbols_spin.value()
            fixed_ratio = self.fixed_ratio_spin.value()
            ttl_days = self.ttl_days_spin.value()
            
            self.logs_text.append("⚙️ Paramètres:")
            self.logs_text.append(f"   - Min symboles: {min_sym}")
            self.logs_text.append(f"   - Max symboles: {max_sym}")
            self.logs_text.append(f"   - Ratio FIXE: {fixed_ratio:.1%} (aléatoire: {1-fixed_ratio:.1%})")
            self.logs_text.append(f"   - Cache TTL: {ttl_days} jours\n")
            
            # Récupérer les symboles d'optimisation
            optim_symbols = [self.optim_list.item(i).text() for i in range(self.optim_list.count())]
            self.logs_text.append(f"📋 Symboles en entrée: {len(optim_symbols)}\n")
            
            if not optim_symbols:
                QMessageBox.warning(self, "Erreur", "Aucun symbole d'optimisation disponible")
                self.logs_text.append("❌ Aucun symbole!")
                return
            
            # Construire la structure secteur×cap
            from symbol_manager import get_all_sectors, get_all_cap_ranges, get_symbols_by_sector_and_cap
            
            sectors = get_all_sectors(list_type="optimization")
            caps = get_all_cap_ranges(list_type="optimization")
            
            self.logs_text.append(f"🏢 Secteurs trouvés: {len(sectors)}")
            self.logs_text.append(f"📊 Cap ranges trouvés: {len(caps)}\n")
            self.logs_text.append("🔨 Construction de la structure secteur×cap...\n")
            
            sector_cap_ranges = {}
            for sec in sectors:
                buckets = {}
                for cap in caps:
                    syms = get_symbols_by_sector_and_cap(sec, cap, list_type="optimization", active_only=True)
                    if syms:
                        buckets[cap] = syms
                if buckets:
                    sector_cap_ranges[sec] = buckets
            
            self.logs_text.append(f"✅ Structure créée: {len(sector_cap_ranges)} secteurs avec groupes\n")
            
            # Lancer le nettoyage hybride
            self.logs_text.append("🧹 Lancement du nettoyage hybride (FIXE + ALÉATOIRE)...\n")
            
            cleaned = clean_sector_cap_groups(
                sector_cap_ranges,
                ttl_days=ttl_days,
                min_symbols=min_sym,
                max_symbols=max_sym,
                fixed_ratio=fixed_ratio
            )
            
            self.logs_text.append("✅ Nettoyage terminé!\n")
            
            # Aplatir et compter
            flat_cleaned = []
            details_lines = []
            
            for sec in sorted(cleaned.keys()):
                for cap in sorted(cleaned[sec].keys()):
                    syms = cleaned[sec][cap]
                    flat_cleaned.extend(syms)
                    preview = ", ".join(syms[:15]) + (" …" if len(syms) > 15 else "")
                    line = f"{sec} × {cap}: {len(syms):3d} → {preview}"
                    details_lines.append(line)
            
            # Affichage résumé
            summary = f"""
═══════════════════════════════════════
       RÉSUMÉ DE L'OPTIMISATION
═══════════════════════════════════════

📊 TOTAL : {len(flat_cleaned)} symboles
   (min garanti: {min_sym}, max: {max_sym})

🔒 Stratégie HYBRIDE:
   - Partie FIXE (mes_symbols): {fixed_ratio:.0%}
   - Partie ALÉATOIRE (popular): {1-fixed_ratio:.0%}

🏢 Groupes traités: {len(cleaned)}
   - Secteurs: {len(cleaned)}
   - Groupes totaux: {sum(len(buckets) for buckets in cleaned.values())}

✅ Statut: Optimisation réussie
"""
            self.summary_text.setText(summary)
            self.details_text.setText("\n".join(details_lines))
            
            self.logs_text.append("="*50)
            self.logs_text.append("✅ OPTIMISATION COMPLÈTE!")
            self.logs_text.append(f"   Total: {len(flat_cleaned)} symboles")
            self.logs_text.append("="*50)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Erreur: {e}\n\n{traceback.format_exc()}"
            self.logs_text.setText(error_msg)
            QMessageBox.critical(self, "Erreur d'optimisation", str(e))
    
    def clear_cache(self):
        """Vider le cache de groupes nettoyés."""
        try:
            from pathlib import Path
            cache_file = Path("cache_data/cleaned_groups_cache.pkl")
            if cache_file.exists():
                cache_file.unlink()
                self.logs_text.append("✅ Cache supprimé")
                QMessageBox.information(self, "Cache", "Cache nettoyé avec succès")
            else:
                self.logs_text.append("ℹ️ Pas de cache à supprimer")
                QMessageBox.information(self, "Cache", "Aucun cache à supprimer")
        except Exception as e:
            self.logs_text.append(f"❌ Erreur: {e}")
            QMessageBox.warning(self, "Erreur", f"Impossible de nettoyer le cache: {e}")
    
    def export_results(self):
        """Exporter les résultats dans un fichier."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.txt"
            
            content = f"""
RÉSULTATS D'OPTIMISATION HYBRIDE
Généré: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PARAMÈTRES:
- Min symboles: {self.min_symbols_spin.value()}
- Max symboles: {self.max_symbols_spin.value()}
- Ratio FIXE: {self.fixed_ratio_spin.value():.0%}
- TTL Cache: {self.ttl_days_spin.value()} jours

RÉSUMÉ:
{self.summary_text.toPlainText()}

DÉTAILS PAR GROUPE:
{self.details_text.toPlainText()}

LOGS:
{self.logs_text.toPlainText()}
"""
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            
            QMessageBox.information(self, "Export", f"Résultats exportés dans: {filename}")
            self.logs_text.append(f"✅ Résultats exportés: {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible d'exporter: {e}")
    
    def copy_to_clipboard(self):
        """Copier la liste des symboles dans le presse-papiers."""
        try:
            symbols = [self.optim_list.item(i).text() for i in range(self.optim_list.count())]
            import pyperclip
            pyperclip.copy("\n".join(symbols))
            QMessageBox.information(self, "Copie", f"{len(symbols)} symboles copiés")
        except ImportError:
            # Fallback si pyperclip non disponible
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            symbols = [self.optim_list.item(i).text() for i in range(self.optim_list.count())]
            clipboard.setText("\n".join(symbols))
            QMessageBox.information(self, "Copie", f"{len(symbols)} symboles copiés")
    
    def showEvent(self, event):
        """Charger les symboles quand la fenêtre s'affiche."""
        super().showEvent(event)
        if self.optim_list.count() == 0:
            self.reload_optimization_symbols()
