import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# Set global matplotlib parameters for larger fonts
plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 27,
    'axes.labelsize': 23,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'figure.titlesize': 35,
    'font.family': 'sans-serif',
    'font.weight': 'normal'
})

def create_folder_structure(base_path):
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'RandomForest'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'SVM'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'MLP'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'KNN'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'Comparison'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'Reports'), exist_ok=True)
    for algo in ['RandomForest', 'SVM', 'MLP', 'KNN']:
        os.makedirs(os.path.join(base_path, algo, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, algo, 'models'), exist_ok=True)
    print("Folder structure created.")

def load_data(train_path, val_path):
    try:
        print(f"Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print("Training data:")
        print(train_df.head())
        print(f"Shape: {train_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")
        print(f"\nLoading validation data from: {val_path}")
        val_df = pd.read_csv(val_path)
        print("Validation data:")
        print(val_df.head())
        print(f"Shape: {val_df.shape}")
        print(f"Columns: {val_df.columns.tolist()}")
        return train_df, val_df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None

def prepare_data(train_df, val_df):
    if train_df is None or val_df is None:
        return None, None, None, None, None, None
    if train_df.isnull().sum().any():
        print("Missing values found in training data. Imputing...")
        train_means = train_df.mean()
        train_df = train_df.fillna(train_means)
    if val_df.isnull().sum().any():
        print("Missing values found in validation data. Imputing...")
        val_df = val_df.fillna(train_means)
    
    # Check for different target column names
    target_col = None
    for col_name in ['Carpeta', 'Folder', 'Label', 'Class']:
        if col_name in train_df.columns and col_name in val_df.columns:
            target_col = col_name
            break
    
    if target_col:
        unique_labels = train_df[target_col].unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_train = train_df[target_col].map(label_map).values
        unknown_labels = set(val_df[target_col].unique()) - set(unique_labels)
        if unknown_labels:
            print(f"WARNING: Found unknown labels in validation set: {unknown_labels}")
            print("Filtering out samples with unknown labels from validation set")
            val_df = val_df[val_df[target_col].isin(unique_labels)]
        y_val = val_df[target_col].map(label_map).values
        
        # Get feature columns (exclude target and image columns)
        exclude_cols = [target_col, 'Imagen', 'Image', 'Filename']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        val_feature_cols = [col for col in val_df.columns if col not in exclude_cols]
        
        if set(feature_cols) != set(val_feature_cols):
            print("WARNING: Feature columns differ between training and validation datasets")
            common_features = list(set(feature_cols).intersection(set(val_feature_cols)))
            print(f"Using only common features: {common_features}")
            feature_cols = common_features
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, y_train, X_val_scaled, y_val, label_map, feature_cols
    else:
        print("Error: Target column not found. Expected 'Carpeta', 'Folder', 'Label', or 'Class'.")
        return None, None, None, None, None, None

def train_evaluate_model(X_train, y_train, X_val, y_val, label_map, feature_cols, model, model_name, base_path):
    if X_train is None or y_train is None or X_val is None or y_val is None:
        return None, None, None, None
    algo_folder_map = {
        "RandomForest": "RandomForest",
        "SVM": "SVM",
        "MLP": "MLP",
        "KNN": "KNN"
    }
    algo_folder = algo_folder_map.get(model_name, model_name)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} validation accuracy: {accuracy:.4f}")
    inv_map = {v: k for k, v in label_map.items()}
    labels = [inv_map[i] for i in range(len(label_map))]
    report = classification_report(y_val, y_pred, target_names=labels, output_dict=True)
    print(classification_report(y_val, y_pred, target_names=labels))
    conf_matrix = confusion_matrix(y_val, y_pred)
    
     # Create confusion matrix with larger fonts
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, 
                annot_kws={'size': 30}, cbar=False)  # Cambiado: size de 14 a 20, agregado cbar=False
    plt.xlabel('Predicted', fontsize=24)
    plt.ylabel('True', fontsize=24)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    img_path = os.path.join(base_path, algo_folder, 'images', f'confusion_matrix_{model_name.lower()}.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    model_path = os.path.join(base_path, algo_folder, 'models', f'{model_name.lower()}_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved at '{model_path}'")
    
    # Create feature importance plot if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(14, 8))
        plt.title(f'Feature Importances - {model_name}', fontsize=18)
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_cols[i] for i in indices], 
                   rotation=90, fontsize=12)
        plt.ylabel('Importance', fontsize=16)
        plt.xlabel('Features', fontsize=16)
        plt.tight_layout()
        importance_path = os.path.join(base_path, algo_folder, 'images', f'feature_importance_{model_name.lower()}.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return model, inv_map, accuracy, report

def generate_roc_curves(models, names, X_val, y_val, inv_map, base_path):
    n_classes = len(np.unique(y_val))
    img_paths = []
    for i in range(n_classes):
        plt.figure(figsize=(12, 9))
        for model, name in zip(models, names):
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_val)
                fpr, tpr, _ = roc_curve(y_val == i, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.title(f'ROC Curve Comparison for class {inv_map[i]}', fontsize=30)
        plt.legend(loc="lower right", fontsize=18)
        plt.grid(alpha=0.3)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        img_path = os.path.join(base_path, 'Comparison', f'roc_comparison_class_{i}.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        img_paths.append(img_path)
        plt.close()
    return img_paths

def compare_models(model_results, base_path):
    names = [r['name'] for r in model_results]
    accuracy = [r['accuracy'] for r in model_results]
    df_comp = pd.DataFrame({
        'Model': names,
        'Validation Accuracy': accuracy
    })
    print("Model Comparison:")
    print(df_comp)
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(names))
    plt.bar(x, accuracy, width=0.5, label='Validation Accuracy')
    plt.xlabel('Model', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Model Performance Comparison', fontsize=18)
    plt.xticks(x, names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(accuracy):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    img_path = os.path.join(base_path, 'Comparison', 'performance_comparison.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    csv_path = os.path.join(base_path, 'Comparison', 'comparison_results.csv')
    df_comp.to_csv(csv_path, index=False)
    print(f"Comparison results saved at '{csv_path}'")
    return img_path, df_comp

def generate_pdf_report(train_path, val_path, model_results, comparison_img_path, roc_paths, base_path):
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
    elements.append(Paragraph("Dataset Information", styles['Heading2']))
    elements.append(Paragraph(f"Training dataset: {os.path.basename(train_path)}", styles['Normal']))
    elements.append(Paragraph(f"Validation dataset: {os.path.basename(val_path)}", styles['Normal']))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph("Results Summary", styles['Heading2']))
    df_comp = pd.DataFrame({
        'Model': [r['name'] for r in model_results],
        'Validation Accuracy': [f"{r['accuracy']:.4f}" for r in model_results]
    })
    data = [df_comp.columns.tolist()] + df_comp.values.tolist()
    table = Table(data, colWidths=[3*inch, 3*inch])
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
        algo_folder_map = {
            'Random Forest': 'RandomForest',
            'SVM': 'SVM',
            'MLP': 'MLP',
            'KNN': 'KNN'
        }
        model_name_map = {
            'Random Forest': 'randomforest',
            'SVM': 'svm',
            'MLP': 'mlp',
            'KNN': 'knn'
        }
        algo_folder = algo_folder_map.get(result['name'], 'RandomForest')
        model_name = model_name_map.get(result['name'], 'randomforest')
        matrix_path = os.path.join(base_path, algo_folder, 'images', f'confusion_matrix_{model_name}.png')
        if os.path.exists(matrix_path):
            elements.append(Paragraph("Confusion Matrix:", styles['Heading3']))
            img_matrix = Image(matrix_path, width=5*inch, height=4*inch)
            elements.append(img_matrix)
            elements.append(Spacer(1, 0.25*inch))
        importance_path = os.path.join(base_path, algo_folder, 'images', f'feature_importance_{model_name}.png')
        if os.path.exists(importance_path):
            elements.append(Paragraph("Feature Importance:", styles['Heading3']))
            img_importance = Image(importance_path, width=5*inch, height=4*inch)
            elements.append(img_importance)
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
        f"Based on the results, the best performing model is {best_model['name']} with an accuracy of {best_model['accuracy']:.4f} on the validation set.",
        styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    doc.build(elements)
    print(f"PDF report generated at '{pdf_path}'")
    return pdf_path

def main():
    train_path = input("Enter the path to the training CSV file: ")
    val_path = input("Enter the path to the validation CSV file: ")
    output_path = input("Enter the path to save outputs (default: './Outputs'): ")
    if not output_path.strip():
        output_path = './Outputs'
    create_folder_structure(output_path)
    train_df, val_df = load_data(train_path, val_path)
    if train_df is not None and val_df is not None:
        X_train, y_train, X_val, y_val, label_map, feature_cols = prepare_data(train_df, val_df)
        if X_train is not None and y_train is not None and X_val is not None and y_val is not None:
            model_results = []
            print("\n" + "="*50)
            print("TRAINING AND EVALUATION: RANDOM FOREST")
            print("="*50)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model, inv_map, rf_acc, rf_report = train_evaluate_model(
                X_train, y_train, X_val, y_val, label_map, feature_cols, rf_model, "RandomForest", output_path)
            model_results.append({
                'name': 'Random Forest',
                'model': rf_model,
                'accuracy': rf_acc,
                'report': rf_report
            })
            print("\n" + "="*50)
            print("TRAINING AND EVALUATION: SVM")
            print("="*50)
            svm_model = SVC(probability=True, random_state=42)
            svm_model, _, svm_acc, svm_report = train_evaluate_model(
                X_train, y_train, X_val, y_val, label_map, feature_cols, svm_model, "SVM", output_path)
            model_results.append({
                'name': 'SVM',
                'model': svm_model,
                'accuracy': svm_acc,
                'report': svm_report
            })
            print("\n" + "="*50)
            print("TRAINING AND EVALUATION: MLP")
            print("="*50)
            mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), 
                                     activation='relu', 
                                     solver='adam', 
                                     alpha=0.0001,
                                     batch_size='auto', 
                                     learning_rate='adaptive',
                                     max_iter=1000, 
                                     random_state=42)
            mlp_model, _, mlp_acc, mlp_report = train_evaluate_model(
                X_train, y_train, X_val, y_val, label_map, feature_cols, mlp_model, "MLP", output_path)
            model_results.append({
                'name': 'MLP',
                'model': mlp_model,
                'accuracy': mlp_acc,
                'report': mlp_report
            })
            print("\n" + "="*50)
            print("TRAINING AND EVALUATION: KNN")
            print("="*50)
            knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
            knn_model, _, knn_acc, knn_report = train_evaluate_model(
                X_train, y_train, X_val, y_val, label_map, feature_cols, knn_model, "KNN", output_path)
            model_results.append({
                'name': 'KNN',
                'model': knn_model,
                'accuracy': knn_acc,
                'report': knn_report
            })
            print("\n" + "="*50)
            print("MODEL COMPARISON")
            print("="*50)
            comparison_img_path, df_comp = compare_models(model_results, output_path)
            print("\nGenerating ROC curves...")
            roc_paths = generate_roc_curves(
                [rf_model, svm_model, mlp_model, knn_model],
                ['Random Forest', 'SVM', 'MLP', 'KNN'],
                X_val, y_val, inv_map,
                output_path)
            generate_pdf_report(
                train_path,
                val_path,
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
        print("Could not load one or both CSV files.")

if __name__ == "__main__":
    main()