import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

def create_folder_structure(base_path):
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'RandomForest'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'SVM'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'Comparison'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'Reports'), exist_ok=True)
    for algo in ['RandomForest', 'SVM']:
        os.makedirs(os.path.join(base_path, algo, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, algo, 'models'), exist_ok=True)
    print("Folder structure created.")

def load_data(path):
    try:
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def prepare_data(df):
    if df is None:
        return None, None, None, None
    if df.isnull().sum().any():
        print("Missing values found. Imputing...")
        df = df.fillna(df.mean())
    if 'Carpeta' in df.columns:
        unique_labels = df['Carpeta'].unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = df['Carpeta'].map(label_map)
        feature_cols = [col for col in df.columns if col not in ['Carpeta', 'Imagen']]
        X = df[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y.values, label_map, feature_cols
    else:
        print("Error: 'Carpeta' column not found.")
        return None, None, None, None

def train_evaluate_model(X, y, label_map, feature_cols, model, model_name, base_path):
    if X is None or y is None:
        return None, None, None, None, None
    algo_folder = "RandomForest" if "RandomForest" in model_name else "SVM"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    inv_map = {v: k for k, v in label_map.items()}
    labels = [inv_map[i] for i in range(len(label_map))]
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=labels))
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    img_path = os.path.join(base_path, algo_folder, 'images', f'confusion_matrix_{model_name.lower()}.png')
    plt.savefig(img_path)
    plt.close()
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{model_name} CV mean: {cv_scores.mean():.4f}, std: {cv_scores.std():.4f}")
    model_path = os.path.join(base_path, algo_folder, 'models', f'{model_name.lower()}_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved at '{model_path}'")
    return model, inv_map, accuracy, cv_scores.mean(), report

def generate_roc_curves(models, names, X, y, inv_map, base_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    n_classes = len(np.unique(y))
    img_paths = []
    for i in range(n_classes):
        plt.figure(figsize=(10, 8))
        for model, name in zip(models, names):
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison for class {inv_map[i]}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        img_path = os.path.join(base_path, 'Comparison', f'roc_comparison_class_{i}.png')
        plt.savefig(img_path)
        img_paths.append(img_path)
        plt.close()
    return img_paths

def compare_models(model_results, base_path):
    names = [r['name'] for r in model_results]
    accuracy = [r['accuracy'] for r in model_results]
    cv_scores = [r['cv_score'] for r in model_results]
    df_comp = pd.DataFrame({
        'Model': names,
        'Test Accuracy': accuracy,
        'Cross-Validation Mean': cv_scores
    })
    print(df_comp)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, accuracy, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_scores, width, label='CV Mean')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(accuracy):
        plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    for i, v in enumerate(cv_scores):
        plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    img_path = os.path.join(base_path, 'Comparison', 'performance_comparison.png')
    plt.savefig(img_path)
    plt.close()
    csv_path = os.path.join(base_path, 'Comparison', 'comparison_results.csv')
    df_comp.to_csv(csv_path, index=False)
    print(f"Comparison results saved at '{csv_path}'")
    return img_path, df_comp

def generate_pdf_report(csv_path, dataset_name, model_results, comparison_img_path, roc_paths, base_path):
    print("Generating PDF report...")
    pdf_path = os.path.join(base_path, 'Reports', 'model_comparison_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Centered', alignment=1))
    elements = []
    elements.append(Paragraph("Model Comparison Report", styles['Title']))
    elements.append(Spacer(1, 0.25*inch))
    date_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    elements.append(Paragraph(f"Date: {date_now}", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"Dataset: {dataset_name}", styles['Normal']))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph("Results Summary", styles['Heading2']))
    df_comp = pd.DataFrame({
        'Model': [r['name'] for r in model_results],
        'Test Accuracy': [f"{r['accuracy']:.4f}" for r in model_results],
        'Cross-Validation': [f"{r['cv_score']:.4f}" for r in model_results]
    })
    data = [df_comp.columns.tolist()] + df_comp.values.tolist()
    table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.25*inch))
    if os.path.exists(comparison_img_path):
        elements.append(Paragraph("Performance Comparison", styles['Heading2']))
        img = Image(comparison_img_path, width=6*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25*inch))
    for result in model_results:
        elements.append(PageBreak())
        elements.append(Paragraph(f"Model Details: {result['name']}", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("Evaluation Metrics:", styles['Heading3']))
        report = result['report']
        metrics_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
        for cls, metrics in report.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_data.append([
                    cls,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}",
                    f"{metrics['support']}"
                ])
        for avg in ['macro avg', 'weighted avg']:
            if avg in report:
                metrics_data.append([
                    avg,
                    f"{report[avg]['precision']:.4f}",
                    f"{report[avg]['recall']:.4f}",
                    f"{report[avg]['f1-score']:.4f}",
                    f"{report[avg]['support']}"
                ])
        metrics_table = Table(metrics_data, colWidths=[1.2*inch]*5)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.25*inch))
        algo_folder = "RandomForest" if "Random Forest" in result['name'] else "SVM"
        model_name = "randomforest" if "Random Forest" in result['name'] else "svm"
        matrix_path = os.path.join(base_path, algo_folder, 'images', f'confusion_matrix_{model_name}.png')
        if os.path.exists(matrix_path):
            elements.append(Paragraph("Confusion Matrix:", styles['Heading3']))
            img_matrix = Image(matrix_path, width=5*inch, height=4*inch)
            elements.append(img_matrix)
            elements.append(Spacer(1, 0.25*inch))
    if roc_paths:
        elements.append(PageBreak())
        elements.append(Paragraph("ROC Curves by Class", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        for path in roc_paths:
            if os.path.exists(path):
                class_name = os.path.basename(path).replace('roc_comparison_class_', '').replace('.png', '')
                elements.append(Paragraph(f"Class {class_name}:", styles['Heading3']))
                img_roc = Image(path, width=5*inch, height=4*inch)
                elements.append(img_roc)
                elements.append(Spacer(1, 0.25*inch))
    elements.append(PageBreak())
    elements.append(Paragraph("Conclusions", styles['Heading2']))
    best_model = max(model_results, key=lambda x: x['accuracy'])
    elements.append(Paragraph(
        f"Based on the results, the best performing model is {best_model['name']} with an accuracy of {best_model['accuracy']:.4f} on the test set and {best_model['cv_score']:.4f} in cross-validation.",
        styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    doc.build(elements)
    print(f"PDF report generated at '{pdf_path}'")
    return pdf_path

def main():
    csv_path = input("Enter the path to the CSV file: ")
    output_path = input("Enter the path to save outputs (default: './Outputs'): ")
    if not output_path.strip():
        output_path = './Outputs'
    create_folder_structure(output_path)
    df = load_data(csv_path)
    if df is not None:
        dataset_name = os.path.basename(csv_path)
        X, y, label_map, feature_cols = prepare_data(df)
        if X is not None and y is not None:
            model_results = []
            print("\n" + "="*50)
            print("TRAINING AND EVALUATION: RANDOM FOREST")
            print("="*50)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model, inv_map, rf_acc, rf_cv, rf_report = train_evaluate_model(
                X, y, label_map, feature_cols, rf_model, "RandomForest", output_path)
            model_results.append({
                'name': 'Random Forest',
                'model': rf_model,
                'accuracy': rf_acc,
                'cv_score': rf_cv,
                'report': rf_report
            })
            print("\n" + "="*50)
            print("TRAINING AND EVALUATION: SVM")
            print("="*50)
            svm_model = SVC(probability=True, random_state=42)
            svm_model, _, svm_acc, svm_cv, svm_report = train_evaluate_model(
                X, y, label_map, feature_cols, svm_model, "SVM", output_path)
            model_results.append({
                'name': 'SVM',
                'model': svm_model,
                'accuracy': svm_acc,
                'cv_score': svm_cv,
                'report': svm_report
            })
            print("\n" + "="*50)
            print("MODEL COMPARISON")
            print("="*50)
            comparison_img_path, df_comp = compare_models(model_results, output_path)
            print("\nGenerating ROC curves...")
            roc_paths = generate_roc_curves(
                [rf_model, svm_model],
                ['Random Forest', 'SVM'],
                X, y, inv_map,
                output_path)
            generate_pdf_report(
                csv_path,
                dataset_name,
                model_results,
                comparison_img_path,
                roc_paths,
                output_path
            )
            print("\nAnalysis completed successfully.")
            print(f"All results and visualizations saved in '{output_path}'.")
        else:
            print("Could not prepare data for analysis.")
    else:
        print("Could not load the CSV file.")

if __name__ == "__main__":
    main()
