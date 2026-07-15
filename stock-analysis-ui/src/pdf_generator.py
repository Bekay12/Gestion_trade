"""
Générateur de rapports PDF professionnels pour l'analyse stock
Gère la génération automatisée des PDFs avec reportlab
"""

import io
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage


class PDFReportGenerator:
    """Générateur de rapports PDF professionnels"""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialiser le générateur
        
        Args:
            results_dir: Chemin du dossier de destination (Results par défaut)
        """
        self.results_dir = results_dir or Path(__file__).parent / "Results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.has_reportlab = self._check_reportlab()
        
    def _check_reportlab(self) -> bool:
        """Vérifier si reportlab est disponible"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Table, Image
            return True
        except ImportError:
            print("⚠️ reportlab non installé, utilisation du fallback matplotlib")
            return False
    
    def export_pdf(self, plots_layout, current_results: List[Dict], clean_columns: List[str], report_meta: Optional[Dict] = None) -> Optional[str]:
        """
        Exporter les graphiques et résultats en PDF
        
        Args:
            plots_layout: Layout contenant les graphiques matplotlib
            current_results: Données des résultats d'analyse
            clean_columns: Colonnes filtrées (sans vides/zéro)
            
        Returns:
            Chemin du PDF créé ou None en cas d'erreur
        """
        report_meta = report_meta or {}
        if self.has_reportlab:
            return self._export_pdf_reportlab(plots_layout, current_results, clean_columns, report_meta)
        else:
            return self._export_pdf_matplotlib(plots_layout, current_results, clean_columns, report_meta)
    
    def _export_pdf_reportlab(self, plots_layout, current_results: List[Dict], 
                             clean_columns: List[str], report_meta: Optional[Dict] = None) -> Optional[str]:
        """Exporter en PDF avec reportlab (professionnel)"""
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm, inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            print("❌ Libraires manquantes, utilisation fallback")
            return self._export_pdf_matplotlib(plots_layout, current_results, clean_columns)
        
        try:
            # Seulement les colonnes internes/redondantes (garder un maximum de données)
            columns_to_skip = {
                '_analysis_id', 'DomaineOriginal', 'ConsensusMean'
            }
            
            # Debug: afficher les colonnes reçues
            print(f"\n📊 GÉNÉRATION PDF - INFO DE DÉBUG")
            print(f"   Colonnes reçues: {len(clean_columns)}")
            print(f"   Colonnes à afficher: {[c for c in clean_columns if c not in columns_to_skip]}")
            print(f"   Résultats: {len(current_results)} symboles\n")
            
            # Générer le nom du fichier
            filename = f"graphiques_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = self.results_dir / filename
            
            # Créer le document en LANDSCAPE
            page_size = landscape(A4)
            doc = SimpleDocTemplate(str(file_path), pagesize=page_size, topMargin=0.5*cm, 
                                   bottomMargin=0.5*cm, leftMargin=0.5*cm, rightMargin=0.5*cm)
            story = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=6,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            canvas_count = 0
            
            # Traiter chaque graphique
            for i in range(plots_layout.count()):
                widget = plots_layout.itemAt(i).widget()
                if widget and hasattr(widget, 'figure'):
                    try:
                        fig_original = widget.figure
                        
                        # Sauvegarder le graphique en haute qualité directement
                        temp_img_path = str(self.results_dir / f"temp_graph_{i}.png")
                        fig_original.savefig(temp_img_path, format='png', dpi=300, 
                                            bbox_inches='tight', facecolor='white',
                                            pad_inches=0.1)
                        
                        # Vérifier que le fichier existe avant de l'utiliser
                        if not Path(temp_img_path).exists():
                            print(f"⚠️ Le fichier temporaire n'a pas pu être créé: {temp_img_path}")
                            continue
                        
                        # Extraire le symbole
                        symbol = None
                        for ax in fig_original.get_axes():
                            title = ax.get_title()
                            if title:
                                parts = title.split('|')
                                if parts:
                                    symbol = parts[0].strip()
                                    break
                        
                        # Trouver les données
                        stock_data = None
                        if symbol:
                            for result in current_results:
                                if result.get('Symbole') == symbol:
                                    stock_data = result
                                    break
                        
                        # Créer le contenu pour cette page
                        page_content = []
                        
                        # Titre
                        title = Paragraph(f"📊 {symbol or 'Graphique'}", title_style)
                        page_content.append(title)
                        page_content.append(Spacer(1, 0.2*cm))
                        
                        # Image (vérifier le chemin) - Pleine largeur landscape
                        try:
                            img_obj = Image(temp_img_path, width=28.5*cm, height=9*cm)
                            page_content.append(img_obj)
                            page_content.append(Spacer(1, 0.3*cm))
                            print(f"✅ Image ajoutée: {temp_img_path}")
                        except Exception as e:
                            print(f"⚠️ Erreur ajout image: {e}")
                        
                        # Infos du tableau - Paramètres sur une ligne, valeurs en dessous
                        if stock_data:
                            # Récupérer TOUTES les colonnes de clean_columns
                            cols_with_data = []
                            for col in clean_columns:
                                if col not in columns_to_skip:
                                    value = stock_data.get(col, 'N/A')
                                    try:
                                        if isinstance(value, float):
                                            formatted_value = f"{value:.2f}"
                                        else:
                                            formatted_value = str(value)
                                    except:
                                        formatted_value = str(value)
                                    cols_with_data.append((col, formatted_value))
                            
                            print(f"   📊 {symbol}: {len(cols_with_data)} colonnes total dans clean_columns")
                            
                            if cols_with_data:
                                # Layout: paramètres sur 1 ligne, valeurs sur la ligne en dessous
                                # Largeur dispo ~28cm en landscape → max 7 colonnes
                                cols_per_group = 7
                                info_data = []
                                
                                for start in range(0, len(cols_with_data), cols_per_group):
                                    group = cols_with_data[start:start + cols_per_group]
                                    # Ligne des noms de paramètres
                                    param_row = [col for col, val in group]
                                    # Ligne des valeurs
                                    value_row = [val for col, val in group]
                                    # Compléter si groupe incomplet
                                    while len(param_row) < cols_per_group:
                                        param_row.append("")
                                        value_row.append("")
                                    info_data.append(param_row)
                                    info_data.append(value_row)
                                
                                if info_data:
                                    col_width = 28.0 * cm / cols_per_group
                                    col_widths = [col_width] * cols_per_group
                                    
                                    info_table = Table(info_data, colWidths=col_widths)
                                    
                                    # Style: alterner header (paramètres) / data (valeurs)
                                    style_cmds = [
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                        ('LEFTPADDING', (0, 0), (-1, -1), 3),
                                        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                                    ]
                                    # Colorer les lignes de paramètres (paires: 0, 2, 4...) en bleu
                                    # et les lignes de valeurs (impaires: 1, 3, 5...) en blanc/gris
                                    for row_idx in range(len(info_data)):
                                        if row_idx % 2 == 0:  # Ligne paramètre
                                            style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#4472C4')))
                                            style_cmds.append(('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.whitesmoke))
                                            style_cmds.append(('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'))
                                            style_cmds.append(('FONTSIZE', (0, row_idx), (-1, row_idx), 7))
                                        else:  # Ligne valeur
                                            style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#F0F8FF')))
                                            style_cmds.append(('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.black))
                                            style_cmds.append(('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica'))
                                            style_cmds.append(('FONTSIZE', (0, row_idx), (-1, row_idx), 7))
                                    
                                    info_table.setStyle(TableStyle(style_cmds))
                                    page_content.append(info_table)
                        
                        # Ajouter le contenu
                        story.append(KeepTogether(page_content))
                        if i < plots_layout.count() - 1:
                            story.append(PageBreak())
                        
                        canvas_count += 1
                        print(f"✅ Graphique {canvas_count} ({symbol or 'inconnu'}) + infos ajoutés au PDF")
                        
                    except Exception as e:
                        print(f"⚠️ Erreur graphique {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Page résumé
            if current_results and canvas_count > 0:
                story.append(PageBreak())

                report_meta = report_meta or {}
                min_holding_days = int(report_meta.get('min_holding_days', 7) or 7)
                
                summary_style = ParagraphStyle(
                    'SummaryTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    textColor=colors.HexColor('#1f4788'),
                    spaceAfter=12,
                    alignment=TA_CENTER,
                    fontName='Helvetica-Bold'
                )
                
                story.append(Paragraph("📊 RÉSUMÉ DE L'ANALYSE", summary_style))
                story.append(Spacer(1, 0.3*cm))
                
                # Statistiques
                total_symbols = len(current_results)
                achats = sum(1 for r in current_results if r.get('Signal') == 'ACHAT')
                ventes = sum(1 for r in current_results if r.get('Signal') == 'VENTE')
                neutres = total_symbols - achats - ventes
                fiabilites = [float(r.get('Fiabilite', 0)) for r in current_results 
                            if isinstance(r.get('Fiabilite'), (int, float, str)) and r.get('Fiabilite') != 'N/A']
                avg_fiabilite = sum(fiabilites)/len(fiabilites) if fiabilites else 0
                gains = [float(r.get('Gain_total', 0.0)) for r in current_results 
                        if isinstance(r.get('Gain_total'), (int, float))]
                gain_total_bt = sum(gains) if gains else 0
                
                # Table stats - Plus large en landscape
                stats_data = [
                    ["Métrique", "Valeur", "Métrique", "Valeur"],
                    ["📈 Total analysé", str(total_symbols), "📅 Date", datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
                    ["✅ Achats", str(achats), "🎯 Fiabilité moy.", f"{avg_fiabilite:.1f}%"],
                    ["❌ Ventes", str(ventes), "💰 Gain total (Backtest)", f"{gain_total_bt:.2f} $"],
                    ["⏱️ Durée min position", f"{min_holding_days} jour(s) actif(s)", "", ""],
                ]
                
                stats_table = Table(stats_data, colWidths=[6*cm, 6*cm, 6*cm, 6*cm])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('TOPPADDING', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8F4F8')]),
                ]))
                
                story.append(stats_table)
                story.append(Spacer(1, 0.5*cm))
                
                # Achats
                story.append(Spacer(1, 0.3*cm))
                story.append(Paragraph("✅ <b>Signaux d'ACHAT</b>", styles['Heading2']))
                achats_data = [["Symbole", "Prix", "Score", "Fiabilité", "Tendance", "RSI", "Domaine", "Gain total"]]
                achats_rows = []
                for r in current_results:
                    if r.get('Signal') == 'ACHAT':
                        fiab = r.get('Fiabilite', 'N/A')
                        try:
                            fiab = f"{float(fiab):.2f}"
                        except (ValueError, TypeError):
                            fiab = str(fiab)
                        prix = r.get('Prix', 0)
                        try:
                            prix_val = float(prix)
                            prix_str = f"{prix_val:.2f}"
                        except (ValueError, TypeError):
                            prix_val = 0
                            prix_str = str(prix)
                        rsi = r.get('RSI', 'N/A')
                        if isinstance(rsi, float):
                            rsi = f"{rsi:.1f}"
                        gain = r.get('Gain_total', 0.0)
                        if isinstance(gain, float):
                            gain = f"{gain:.2f}"
                        achats_rows.append((prix_val, [
                            r.get('Symbole', 'N/A'),
                            prix_str,
                            str(r.get('Score', 'N/A')), 
                            fiab,
                            str(r.get('Tendance', 'N/A')),
                            str(rsi),
                            str(r.get('Domaine', 'N/A')),
                            str(gain)
                        ]))
                # Trier par prix décroissant
                achats_rows.sort(key=lambda x: x[0], reverse=True)
                for _, row in achats_rows:
                    achats_data.append(row)
                
                if len(achats_data) > 1:
                    achats_table = Table(achats_data, colWidths=[2.8*cm, 2.5*cm, 2.2*cm, 2.5*cm, 2.8*cm, 2.2*cm, 4.5*cm, 2.8*cm])
                    achats_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00B050')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#E8F5E9'), colors.white]),
                    ]))
                    story.append(achats_table)
                
                # Ventes
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph("❌ <b>Signaux de VENTE</b>", styles['Heading2']))
                ventes_data = [["Symbole", "Prix", "Score", "Fiabilité", "Tendance", "RSI", "Domaine", "Gain total"]]
                ventes_rows = []
                for r in current_results:
                    if r.get('Signal') == 'VENTE':
                        fiab = r.get('Fiabilite', 'N/A')
                        try:
                            fiab = f"{float(fiab):.2f}"
                        except (ValueError, TypeError):
                            fiab = str(fiab)
                        prix = r.get('Prix', 0)
                        try:
                            prix_val = float(prix)
                            prix_str = f"{prix_val:.2f}"
                        except (ValueError, TypeError):
                            prix_val = 0
                            prix_str = str(prix)
                        rsi = r.get('RSI', 'N/A')
                        if isinstance(rsi, float):
                            rsi = f"{rsi:.1f}"
                        gain = r.get('Gain_total', 0.0)
                        if isinstance(gain, float):
                            gain = f"{gain:.2f}"
                        ventes_rows.append((prix_val, [
                            r.get('Symbole', 'N/A'),
                            prix_str,
                            str(r.get('Score', 'N/A')), 
                            fiab,
                            str(r.get('Tendance', 'N/A')),
                            str(rsi),
                            str(r.get('Domaine', 'N/A')),
                            str(gain)
                        ]))
                # Trier par prix décroissant
                ventes_rows.sort(key=lambda x: x[0], reverse=True)
                for _, row in ventes_rows:
                    ventes_data.append(row)
                
                if len(ventes_data) > 1:
                    ventes_table = Table(ventes_data, colWidths=[2.8*cm, 2.5*cm, 2.2*cm, 2.5*cm, 2.8*cm, 2.2*cm, 4.5*cm, 2.8*cm])
                    ventes_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF0000')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFEBEE'), colors.white]),
                    ]))
                    story.append(ventes_table)
            
            # Construire le PDF
            doc.build(story)
            
            print(f"✅ PDF professionnel créé: {file_path}")
            return str(file_path)
            
        except Exception as e:
            print(f"❌ Erreur export PDF reportlab: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _export_pdf_matplotlib(self, plots_layout, current_results: List[Dict], 
                              clean_columns: List[str], report_meta: Optional[Dict] = None) -> Optional[str]:
        """Fallback: Exporter en PDF avec matplotlib"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        try:
            columns_to_skip = {
                'Signal', 'Score', 'Prix', 'Tendance', 'RSI', 
                'Volume moyen', 'Consensus', '_analysis_id',
                'DomaineOriginal', 'ConsensusMean', 'Symbole'
            }
            
            filename = f"graphiques_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = self.results_dir / filename
            
            with PdfPages(str(file_path)) as pdf:
                canvas_count = 0
                
                for i in range(plots_layout.count()):
                    widget = plots_layout.itemAt(i).widget()
                    if widget and hasattr(widget, 'figure'):
                        try:
                            fig_original = widget.figure
                            pdf.savefig(fig_original, bbox_inches='tight')
                            canvas_count += 1
                            
                            # Extraire symbole
                            symbol = None
                            for ax in fig_original.get_axes():
                                title = ax.get_title()
                                if title:
                                    symbol = title.split('|')[0].strip()
                                    break
                            
                            # Page d'infos
                            if symbol:
                                stock_data = None
                                for result in current_results:
                                    if result.get('Symbole') == symbol:
                                        stock_data = result
                                        break
                                
                                if stock_data:
                                    fig_info = plt.figure(figsize=(11, 8.5))
                                    ax_info = fig_info.add_subplot(111)
                                    ax_info.axis('off')
                                    
                                    info_lines = [f"📋 INFORMATIONS DÉTAILLÉES - {symbol}", "=" * 100, ""]
                                    
                                    for col in clean_columns:
                                        if col not in columns_to_skip:
                                            value = stock_data.get(col, '')
                                            if value and value != '' and value != 0 and value != '0':
                                                if isinstance(value, float):
                                                    formatted_value = f"{value:.2f}"
                                                else:
                                                    formatted_value = str(value)
                                                info_lines.append(f"  • {col}: {formatted_value}")
                                    
                                    info_text = "\n".join(info_lines)
                                    ax_info.text(0.05, 0.95, info_text,
                                               transform=ax_info.transAxes,
                                               fontfamily='monospace',
                                               fontsize=9,
                                               verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
                                    
                                    pdf.savefig(fig_info, bbox_inches='tight')
                                    plt.close(fig_info)
                            
                            print(f"✅ Graphique {canvas_count} ({symbol or 'inconnu'}) + infos ajoutés")
                        except Exception as e:
                            print(f"⚠️ Erreur graphique {i}: {e}")
                            continue
                
                # Résumé
                if current_results:
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis('off')

                    report_meta = report_meta or {}
                    min_holding_days = int(report_meta.get('min_holding_days', 7) or 7)
                    
                    total_symbols = len(current_results)
                    achats = sum(1 for r in current_results if r.get('Signal') == 'ACHAT')
                    ventes = sum(1 for r in current_results if r.get('Signal') == 'VENTE')
                    fiabilites = [float(r.get('Fiabilite', 0)) for r in current_results 
                                if isinstance(r.get('Fiabilite'), (int, float, str)) and r.get('Fiabilite') != 'N/A']
                    
                    title_text = "📊 RÉSUMÉ DE L'ANALYSE STOCK\n"
                    title_text += f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
                    
                    stats_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║               STATISTIQUES GLOBALES                           ║
╚═══════════════════════════════════════════════════════════════╝

📈 Total de symboles analysés: {total_symbols}
✅ Signaux ACHAT: {achats}
❌ Signaux VENTE: {ventes}
⚪ Signaux NEUTRE: {total_symbols - achats - ventes}

{"🎯 Fiabilité moyenne: " + f"{sum(fiabilites)/len(fiabilites):.1f}%" if fiabilites else "🎯 Fiabilité moyenne: N/A"}
⏱️ Durée min position: {min_holding_days} jour(s) actif(s)
"""
                    
                    ax.text(0.05, 0.95, title_text + stats_text, transform=ax.transAxes, 
                           fontfamily='monospace', fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                
                d = pdf.infodict()
                d['Title'] = 'Analyse Stock'
                d['Author'] = 'Stock Analysis Tool'
                d['Subject'] = 'Résultats d\'analyse technique'
                d['Keywords'] = 'Stock Analysis'
                d['CreationDate'] = datetime.now()
            
            print(f"✅ PDF matplotlib créé: {file_path}")
            return str(file_path)
            
        except Exception as e:
            print(f"❌ Erreur export PDF matplotlib: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    print("Module de génération de rapports PDF")
    print("À utiliser: from pdf_generator import PDFReportGenerator")
