import pandas as pd

# Load both datasets
xlsr_df = pd.read_csv("C:/Users/lenovo/Desktop/btpcodeeeeee/sarcasm-detection/audio_features_xlsr.csv")
imagebind_df = pd.read_csv("C:/Users/lenovo/Desktop/btpcodeeeeee/sarcasm-detection/audio_features_lb.csv")

# Ensure both datasets have a common column like 'ID' or 'filename'
common_column = "ID"  # Update this if the column name is different

# Merge datasets on the common identifier
merged_df = pd.merge(xlsr_df, imagebind_df, on=common_column, how="inner")

# Identify feature types
audio_context_cols = [col for col in merged_df.columns if "audio_c_feature_" in col]
audio_utterance_cols = [col for col in merged_df.columns if "audio_u_feature_" in col]
metadata_cols = ["Sarcasm", "Sarcasm_Type", "Implicit_Emotion", "Explicit_Emotion", "Valence", "Arousal"]

# Ensure the correct order: ID → audio_context → audio_utterance → metadata/labels
final_cols = [common_column] + audio_context_cols + audio_utterance_cols + metadata_cols

# Reorder dataframe
merged_df = merged_df[final_cols]

# Save the merged dataset
merged_df.to_csv("merged_features.csv", index=False)

print("Merged dataset saved as 'merged_features.csv'. Shape:", merged_df.shape)
