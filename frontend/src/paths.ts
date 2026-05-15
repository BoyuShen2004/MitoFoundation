/** Project-relative paths (resolved server-side under MITO2_PROJECT_ROOT). */

export const REL_DATA_RAW = "data/raw";
export const REL_NNUNET_DATASET = "data/nnUNet_raw/Dataset001_mito2";
export const REL_OUTPUTS_BC = "data/outputs/bc";
export const REL_OUTPUTS_POSTPROCESSED = "data/outputs/postprocessed";
export const REL_NNUNET_LABELS_TS_INSTANCE = `${REL_NNUNET_DATASET}/labelsTs-instance`;

/** Default until MitoLE config API returns base_path. */
export const DEFAULT_MITOLE_BASE_PATH = "/projects/weilab/dataset/MitoLE";
