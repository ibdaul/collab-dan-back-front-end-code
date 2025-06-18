from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


MODEL_PATH = 'model/'
try:
    kmeans_model = joblib.load(MODEL_PATH + 'kmeans.joblib')
    scaler = joblib.load(MODEL_PATH + 'standard_scaler.joblib')
    label_encoders = joblib.load(MODEL_PATH + 'label_encoders.joblib')
    feature_names_for_model = joblib.load(MODEL_PATH + 'feature_names_for_model.joblib') 
    categorical_cols = joblib.load(MODEL_PATH + 'categorical_cols.joblib')
    
    app.logger.info(f"Scaler feature_names_in_: {getattr(scaler, 'feature_names_in_', 'Tidak tersedia')}")
    app.logger.info(f"Scaler n_features_in_: {getattr(scaler, 'n_features_in_', 'Tidak tersedia')}")
    app.logger.info(f"Feature names for model (KMeans) from joblib: {feature_names_for_model}")
    app.logger.info(f"Categorical columns from joblib: {categorical_cols}")

except FileNotFoundError as e:
    app.logger.error(f"Error memuat file model: {e}. Pastikan semua file .joblib ada di direktori {MODEL_PATH}")
    raise e

# --- MODIFIKASI DI SINI ---
# Mengubah pemetaan cluster menjadi 3 kategori: Rendah, Sedang, Tinggi.
cluster_labels_map_backend = {
    0: "Risiko Rendah",
    1: "Risiko Sedang",
    2: "Risiko Tinggi"
}

numerical_cols_for_this_scaler = [
    'network_packet_size', 'login_attempts', 'session_duration',
    'ip_reputation_score', 'failed_logins', 'unusual_time_access'
]


def preprocess_input_single_row(data_dict):
    processed_values = {}

    for col in feature_names_for_model:
        value_from_input = data_dict.get(col, None) # Gunakan .get() untuk default ke None jika kolom hilang

        if col in categorical_cols:
            le = label_encoders.get(col)
            if not le:
                raise ValueError(f"LabelEncoder untuk kolom '{col}' tidak ditemukan.")
            
            value_for_le_transform = None
            
            # Menangani nilai yang hilang/kosong terlebih dahulu
            is_missing_input = False
            if pd.isna(value_from_input): # Menangani Python None, np.nan
                is_missing_input = True
            elif isinstance(value_from_input, str) and value_from_input.lower() in ['tidak ada', 'none', '', 'nan']:
                is_missing_input = True

            if is_missing_input:
                # Untuk LabelEncoder yang dilatih dengan np.nan, kita harus memberikan np.nan
                value_for_le_transform = np.nan
                app.logger.info(f"Input '{value_from_input}' untuk kolom kategorikal '{col}' dipetakan ke np.nan untuk transform.")
            elif col == 'browser_type': # Penanganan spesifik setelah dipastikan bukan missing
                str_val_lower = str(value_from_input).lower()
                if str_val_lower == 'lainnya' or str_val_lower == 'unknown':
                    if 'Unknown' in le.classes_:
                        value_for_le_transform = 'Unknown'
                    else: 
                        value_for_le_transform = str(value_from_input) 
                        app.logger.warning(f"Kategori '{value_from_input}' (Lainnya/Unknown) untuk '{col}' akan dicoba transform langsung. Kelas: {list(le.classes_)}")
                else:
                    value_for_le_transform = str(value_from_input)
            else: # Default untuk kolom kategorikal lain yang tidak missing
                value_for_le_transform = str(value_from_input)
            
            try:
                # LabelEncoder.transform() mengharapkan array/list
                encoded_val = le.transform([value_for_le_transform])[0]
                processed_values[col] = int(encoded_val)
            except ValueError as e:
                error_msg = (f"Nilai '{value_for_le_transform}' (dari input asli '{value_from_input}') untuk kolom '{col}' tidak dikenal LabelEncoder. "
                             f"Kelas yang dikenal: {list(le.classes_)}. Error asli: {e}")
                app.logger.error(error_msg)
                # Jika transformasi gagal (misalnya, np.nan tidak ada di kelas encoder, atau nilai tak terduga lain)
                # Isi dengan NaN yang akan diimputasi nanti sebelum konversi tipe.
                processed_values[col] = np.nan 
                app.logger.warning(f"Kolom '{col}' diisi NaN karena transformasi LabelEncoder gagal.")
        
        else: # Kolom numerik
            num_val = pd.to_numeric(value_from_input, errors='coerce')
            if pd.isna(num_val):
                # Imputasi NaN untuk kolom numerik. Harus konsisten dengan training.
                # Mengisi dengan 0.0 sebagai contoh.
                app.logger.warning(f"Kolom numerik '{col}' tidak valid (nilai: '{value_from_input}'). Diisi dengan 0.0.")
                processed_values[col] = 0.0 
            else:
                processed_values[col] = float(num_val) # Pastikan float untuk konsistensi

    # Membuat DataFrame dari dictionary nilai yang sudah diproses
    # Ini memastikan urutan kolom sesuai feature_names_for_model
    df_for_kmeans = pd.DataFrame([processed_values], columns=feature_names_for_model)

    # Paksa tipe data setelah semua nilai diisi dan sebelum scaling
    for col_name in feature_names_for_model:
        if col_name in categorical_cols:
            if df_for_kmeans[col_name].isnull().any():
                app.logger.warning(f"Kolom kategorikal '{col_name}' masih NaN setelah encoding. Mengisi dengan 0 sebelum konversi ke int.")
                df_for_kmeans[col_name].fillna(0, inplace=True) # Atau imputasi lain (misal, modus)
            df_for_kmeans[col_name] = df_for_kmeans[col_name].astype(int)
        else: # Kolom numerik (baik yang akan di-scale maupun tidak jika ada)
            if df_for_kmeans[col_name].isnull().any(): # Seharusnya sudah diimputasi di atas
                app.logger.warning(f"Kolom numerik '{col_name}' masih NaN. Mengisi dengan 0.0 sebelum konversi ke float.")
                df_for_kmeans[col_name].fillna(0.0, inplace=True)
            df_for_kmeans[col_name] = df_for_kmeans[col_name].astype(float)

    app.logger.info(f"Data types SEBELUM scaling subset: \n{df_for_kmeans.dtypes}")
    app.logger.info(f"Data sample SEBELUM scaling subset: \n{df_for_kmeans.head()}")

    # Scaling hanya pada kolom numerik yang diharapkan oleh scaler saat ini
    df_subset_to_scale = df_for_kmeans[numerical_cols_for_this_scaler].copy()

    if df_subset_to_scale.isnull().any().any():
        app.logger.error(f"KRITIS: NaN ditemukan di subset untuk scaling: \n{df_subset_to_scale[df_subset_to_scale.isnull().any(axis=1)]}")
        # Imputasi darurat jika masih ada NaN (seharusnya tidak jika imputasi di atas bekerja)
        for col_to_impute in df_subset_to_scale.columns[df_subset_to_scale.isnull().any()]:
            app.logger.warning(f"Mengisi NaN (darurat) di kolom {col_to_impute} dengan 0.0 sebelum scaling.")
            df_subset_to_scale[col_to_impute].fillna(0.0, inplace=True)

    try:
        scaled_values_subset = scaler.transform(df_subset_to_scale)
        for i, scaled_col_name in enumerate(numerical_cols_for_this_scaler):
            df_for_kmeans.loc[0, scaled_col_name] = scaled_values_subset[0, i]
    except Exception as e:
        # ... (error handling scaling seperti sebelumnya) ...
        app.logger.error(f"Error saat scaling subset. Scaler feature_names_in: {getattr(scaler, 'feature_names_in_', 'N/A')}")
        app.logger.error(f"Kolom subset yang diberikan ke scaler: {df_subset_to_scale.columns.tolist()}")
        app.logger.error(f"Data types subset yang diberikan: \n{df_subset_to_scale.dtypes}")
        app.logger.error(f"Data subset sebelum scaling:\n{df_subset_to_scale}. Error: {e}")
        raise ValueError(f"Error saat scaling data subset: {e}")
            
    # Pastikan semua kolom di df_for_kmeans adalah numerik sebelum dikembalikan
    for col_final_check in df_for_kmeans.columns:
        if df_for_kmeans[col_final_check].dtype == 'object':
            app.logger.error(f"Kolom '{col_final_check}' masih object SETELAH scaling. Mengkonversi paksa ke float, mengisi NaN dengan 0.")
            df_for_kmeans[col_final_check] = pd.to_numeric(df_for_kmeans[col_final_check], errors='coerce').fillna(0).astype(float)
        elif pd.api.types.is_integer_dtype(df_for_kmeans[col_final_check]) and col_final_check not in categorical_cols:
            # Jika kolom numerik menjadi integer (misal unusual_time_access), konversi ke float untuk konsistensi dengan scaler
            df_for_kmeans[col_final_check] = df_for_kmeans[col_final_check].astype(float)


    app.logger.info(f"Data types FINAL untuk KMeans (df_for_kmeans): \n{df_for_kmeans.dtypes}")
    return df_for_kmeans # Harusnya sudah sesuai dengan feature_names_for_model


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Tidak ada data input"}), 400

        if isinstance(input_data, list): 
            if not input_data:
                return jsonify({"error": "Data CSV kosong"}), 400
            
            processed_data_for_prediction_list = []
            original_data_with_errors = [] 

            for i, row_dict in enumerate(input_data):
                try:
                    # current_row_data diisi dengan nilai dari row_dict, default None jika kolom tidak ada
                    current_row_data = {expected_col: row_dict.get(expected_col) for expected_col in feature_names_for_model}
                    df_processed_single = preprocess_input_single_row(current_row_data)
                    processed_data_for_prediction_list.append(df_processed_single.iloc[0])
                except ValueError as ve_row:
                    app.logger.warning(f"Baris data CSV ke-{i+1} dilewati karena error preprocessing: {ve_row}. Data baris: {row_dict}")
                    original_data_with_errors.append({"row_index": i+1, "data": row_dict, "error": str(ve_row)}) 
                    continue 
            
            if not processed_data_for_prediction_list:
                 return jsonify({"error": "Tidak ada data valid yang bisa diproses dari CSV setelah preprocessing.", "row_errors": original_data_with_errors}), 400
            
            all_processed_df = pd.DataFrame(processed_data_for_prediction_list, columns=feature_names_for_model)
            
            app.logger.info(f"DataFrame yang akan diprediksi (dari CSV, {all_processed_df.shape[0]} baris):\n{all_processed_df.head()}")
            app.logger.info(f"Tipe data DataFrame yang akan diprediksi:\n{all_processed_df.dtypes}")

            # Pastikan semua kolom adalah numerik sebelum ke KMeans
            for col in all_processed_df.columns:
                 if not pd.api.types.is_numeric_dtype(all_processed_df[col]):
                    app.logger.error(f"Kolom '{col}' di DataFrame gabungan masih bertipe '{all_processed_df[col].dtype}' sebelum prediksi KMeans. Harusnya numerik.")
                    # Ini seharusnya sudah ditangani oleh preprocess_input_single_row dan type casting finalnya
                    # Jika masih terjadi, ada masalah di logika type casting di preprocess_input_single_row
                    # Tambahkan konversi paksa di sini jika perlu, tapi idealnya tidak.
                    # all_processed_df[col] = pd.to_numeric(all_processed_df[col], errors='coerce').fillna(0)


            predictions = kmeans_model.predict(all_processed_df)
            
            response_payload = {"clusters": [int(p) for p in predictions]}
            if original_data_with_errors:
                response_payload["processed_rows_count"] = len(processed_data_for_prediction_list)
                response_payload["skipped_rows_count"] = len(original_data_with_errors)
                response_payload["row_errors"] = original_data_with_errors
            return jsonify(response_payload)
        
        elif isinstance(input_data, dict): 
            current_input_data = {expected_col: input_data.get(expected_col) for expected_col in feature_names_for_model}
            df_processed = preprocess_input_single_row(current_input_data)
            
            app.logger.info(f"DataFrame yang akan diprediksi (manual):\n{df_processed.head()}")
            app.logger.info(f"Tipe data DataFrame yang akan diprediksi:\n{df_processed.dtypes}")

            # Pastikan semua kolom adalah numerik sebelum ke KMeans
            for col in df_processed.columns:
                if not pd.api.types.is_numeric_dtype(df_processed[col]):
                    app.logger.error(f"Kolom '{col}' di DataFrame (manual) masih bertipe '{df_processed[col].dtype}' sebelum prediksi KMeans.")


            prediction = kmeans_model.predict(df_processed)
            cluster_id = int(prediction[0])
            
            return jsonify({
                "cluster_id": cluster_id,
                "cluster_label": cluster_labels_map_backend.get(cluster_id, f"Cluster {cluster_id} (Label Tidak Didefinisikan)")
            })
        else:
            return jsonify({"error": "Format input tidak didukung. Harap kirim JSON object atau array of objects."}), 400

    except ValueError as ve:
        app.logger.error(f"ValueError dalam API /predict: {ve}", exc_info=True)
        return jsonify({"error": f"Kesalahan data input atau preprocessing: {str(ve)}"}), 400
    except Exception as e:
        app.logger.error(f"Error pada endpoint /predict: {e}", exc_info=True)
        return jsonify({"error": f"Terjadi kesalahan internal server: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)