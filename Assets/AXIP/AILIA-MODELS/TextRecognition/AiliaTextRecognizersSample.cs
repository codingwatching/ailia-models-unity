using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Runtime.InteropServices;

using UnityEngine;
using UnityEngine.UI;

using ailia;

namespace ailiaSDK
{
    public class AiliaTextRecognizersSample : AiliaRenderer
    {
        public enum TextRecognizerModels
        {
            PaddleOCRV1,
            PaddleOCRV3,
            Debug
        }

        public enum Language
        {
            Japanese,
            English,
            Chinese,
            German,
            French,
            Korean
        }

        public enum ModelSize
        {
            Server,
            Mobile
        }

        public enum OutputMode
        {
            DetectedRoi,
            RecognizedText
        }

        [SerializeField]
        private TextRecognizerModels ailiaModelType = TextRecognizerModels.PaddleOCRV1;
        [SerializeField]
        private Language language = Language.Japanese;
        [SerializeField]
        private ModelSize modelSize = ModelSize.Server;
        [SerializeField]
        private OutputMode output_mode = OutputMode.DetectedRoi;
        [SerializeField]
        private GameObject UICanvas = null;

        //Settings
        public bool gpu_mode = false;
        public bool video_mode = false;
        public int camera_id = 0;
        public bool debug = false;
        public Texture2D test_image = null;

        //Result
        public Text label_text = null;
        public Text mode_text = null;
        public RawImage raw_image = null;
        public GameObject text_mesh = null;

        //Preview
        private Texture2D preview_texture = null;

        //AILIA
        private AiliaCamera ailia_camera = new AiliaCamera();
        private AiliaDownload ailia_download = new AiliaDownload();

        // AILIA open file
        private AiliaModel ailia_text_detector = new AiliaModel();
        private AiliaModel ailia_text_classificator = new AiliaModel();
        private AiliaModel ailia_text_recognizer = new AiliaModel();

        private AiliaPaddleOCR paddle_ocr = new AiliaPaddleOCR();

        private String[] txt_file;

        private float UIImageWidth;
        private float UIImageHeight;

        private bool FileOpened = false;

        private void CreateAiliaTextRecognizer()
        {
            string asset_path = Application.temporaryCachePath;
            var urlList = new List<ModelDownloadURL>();

            if (gpu_mode)
            {
                ailia_text_recognizer.Environment(Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
            }

            var dict_path = "";

            var weight_path_detection = "";
            var weight_path_classification = "";
            var weight_path_recognition = "";

            var model_path_detection = "";
            var model_path_classification = "";
            var model_path_recognition = "";

            switch (ailiaModelType)
            {
                case TextRecognizerModels.PaddleOCRV1:
                    mode_text.text = "ailia text Recognizer (V1)";

                    weight_path_detection = "chi_eng_num_sym_server_det_org.onnx";
                    weight_path_classification = "chi_eng_num_sym_mobile_cls_org.onnx";
                    weight_path_recognition = "";

                    switch (language)
                    {
                        case Language.Japanese:
                            if (modelSize == ModelSize.Server){
                                weight_path_recognition = "jpn_eng_num_sym_server_rec_add.onnx";
                                dict_path = "jpn_eng_num_sym_add.txt";
                            }else{
                                weight_path_recognition = "jpn_eng_num_sym_mobile_rec_org.onnx";
                                dict_path = "jpn_eng_num_sym_org.txt";
                            }
                            break;
                        case Language.English:
                            weight_path_recognition = "eng_num_sym_mobile_rec_org.onnx";
                            dict_path = "eng_num_sym_org.txt";
                            break;
                        case Language.Chinese:
                            if (modelSize == ModelSize.Server){
                                weight_path_recognition = "chi_eng_num_sym_server_rec_org.onnx";
                                dict_path = "chi_eng_num_sym_org.txt";
                            }else{
                                weight_path_recognition = "chi_eng_num_sym_mobile_rec_org.onnx";
                                dict_path = "chi_eng_num_sym_org.txt";
                            }
                            break;
                        case Language.German:
                            weight_path_recognition = "ger_eng_num_sym_mobile_rec_org.onnx";
                            dict_path = "ger_eng_num_sym_org.txt";
                            break;
                        case Language.French:
                            weight_path_recognition = "fre_eng_num_sym_mobile_rec_org.onnx";
                            dict_path = "fre_eng_num_sym_org.txt";
                            break;
                        case Language.Korean:
                            weight_path_recognition = "kor_eng_num_sym_mobile_rec_org.onnx";
                            dict_path = "kor_eng_num_sym_org.txt";
                            break;
                        default:
                            Debug.Log("Others language are working in progress.");
                            break;
                    }

                    model_path_detection = weight_path_detection + ".prototxt";
                    model_path_classification = weight_path_classification + ".prototxt";
                    model_path_recognition = weight_path_recognition + ".prototxt";

                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = model_path_detection });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = weight_path_detection });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = model_path_classification });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = weight_path_classification });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = model_path_recognition });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = weight_path_recognition });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = dict_path });

                    paddle_ocr.SetVersion(1);

                    StartCoroutine(ailia_download.DownloadWithProgressFromURL(urlList, () =>
                    {
                        FileOpened = ailia_text_detector.OpenFile(asset_path + "/" + model_path_detection, asset_path + "/" + weight_path_detection);
                        FileOpened = ailia_text_classificator.OpenFile(asset_path + "/" + model_path_classification, asset_path + "/" + weight_path_classification);
                        FileOpened = ailia_text_recognizer.OpenFile(asset_path + "/" + model_path_recognition, asset_path + "/" + weight_path_recognition);
                    }));
                    break;

                case TextRecognizerModels.PaddleOCRV3:
                    mode_text.text = "ailia text Recognizer (V3)";
                    dict_path = "PP-OCRv5_rec.txt";

                    weight_path_detection = "PP-OCRv5_mobile_det_infer.onnx";
                    weight_path_classification = "chi_eng_num_sym_mobile_cls_org.onnx";
                    weight_path_recognition = "PP-OCRv5_mobile_rec_infer.onnx";

                    if (modelSize == ModelSize.Server){
                        weight_path_detection = "PP-OCRv5_server_det_infer.onnx";
                        weight_path_recognition = "PP-OCRv5_server_rec_infer.onnx";
                    }

                    model_path_detection = weight_path_detection + ".prototxt";
                    model_path_classification = weight_path_classification + ".prototxt";
                    model_path_recognition = weight_path_recognition + ".prototxt";

                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr_v3", file_name = model_path_detection });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr_v3", file_name = weight_path_detection });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = model_path_classification });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr", file_name = weight_path_classification });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr_v3", file_name = model_path_recognition });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr_v3", file_name = weight_path_recognition });
                    urlList.Add(new ModelDownloadURL() { folder_path = "paddle_ocr_v3", file_name = dict_path });

                    paddle_ocr.SetVersion(3);

                    StartCoroutine(ailia_download.DownloadWithProgressFromURL(urlList, () =>
                    {
                        LoadDictionary(asset_path, dict_path);
                        FileOpened = ailia_text_detector.OpenFile(asset_path + "/" + model_path_detection, asset_path + "/" + weight_path_detection);
                        FileOpened = ailia_text_classificator.OpenFile(asset_path + "/" + model_path_classification, asset_path + "/" + weight_path_classification);
                        FileOpened = ailia_text_recognizer.OpenFile(asset_path + "/" + model_path_recognition, asset_path + "/" + weight_path_recognition);
                    }));
                    break;

                default:
                    Debug.Log("Others ailia models are working in progress.");
                    break;
            }
        }

        private void LoadDictionary(string asset_path, string dict_path){
            //辞書ファイルの読み込み
            txt_file = File.ReadAllLines(asset_path + "/" + dict_path);
            string[] tmp = new string[txt_file.Length + 1]; //先頭に'blank'を追加
            tmp[0] = "blank";
            for (int i = 0; i < txt_file.Length; i++)
            {
                tmp[i + 1] = txt_file[i];
            }
            txt_file = tmp;
        }

        private void DestroyAiliaDetector()
        {
            ailia_text_detector.Close();
            ailia_text_classificator.Close();
            ailia_text_recognizer.Close();
        }


        void Start()
        {
			AiliaLicense.CheckAndDownloadLicense();
            SetUIProperties();
            CreateAiliaTextRecognizer();
            ailia_camera.CreateCamera(camera_id, false);

            UIImageWidth = raw_image.rectTransform.rect.size.x; //800
            UIImageHeight = raw_image.rectTransform.rect.size.y; //800
        }


        void Update()
        {
            if (!ailia_camera.IsEnable())
            {
                return;
            }
            if (!FileOpened)
            {
                return;
            }

            //Clear result
            Clear();

            //Get camera image
            Color32[] camera = ailia_camera.GetPixels32();
            int original_width = ailia_camera.GetWidth();
            int original_height = ailia_camera.GetHeight();      

            if (!video_mode)
            {
                camera = test_image.GetPixels32();
                original_width = test_image.width;
                original_height = test_image.height;
            }

            int tex_width = paddle_ocr.PADDLEOCR_DETECTOR_INPUT_WIDTH_SIZE; //1536;
            int tex_height = paddle_ocr.PADDLEOCR_DETECTOR_INPUT_HEIGHT_SIZE; //839;

            if(camera.Length != tex_width * tex_height){
                camera = ResizeColorArray(camera, original_width, original_height, tex_width, tex_height);
            }
            
            //Predict
            long start_time = DateTime.UtcNow.Ticks / TimeSpan.TicksPerMillisecond;
            List<AiliaPaddleOCR.TextInfo> result_detections = paddle_ocr.Detection(ailia_text_detector, camera, tex_width, tex_height);
            List<AiliaPaddleOCR.TextInfo> result_classifications = paddle_ocr.Classification(ailia_text_recognizer, camera, tex_width, tex_height, result_detections);
            List<AiliaPaddleOCR.TextInfo> result_recognitions = paddle_ocr.Recognition(ailia_text_recognizer, camera, tex_width, tex_height, result_classifications, txt_file, language, modelSize);
            long end_time = DateTime.UtcNow.Ticks / TimeSpan.TicksPerMillisecond;
            long recognition_time = (end_time - start_time);
        

            //Draw result
            if (ailiaModelType == TextRecognizerModels.PaddleOCRV1 || ailiaModelType == TextRecognizerModels.PaddleOCRV3)
            {
                float ratio = tex_height/(float)tex_width;
                raw_image.rectTransform.sizeDelta = new Vector2(UIImageWidth, UIImageHeight * ratio);

                // Input image preview
                if (preview_texture == null)
                {
                    preview_texture = new Texture2D(tex_width, tex_height);
                    raw_image.texture = preview_texture;
                    raw_image.color = new Color32(128,128,128,255);
                }

                // Apply
                preview_texture.SetPixels32(camera);
                preview_texture.Apply();

                if(result_recognitions == null){
                    return;
                }

                // Detected roi or text
                if(output_mode == OutputMode.DetectedRoi){
                    for (int i = 0; i < result_recognitions.Count; i++)
                    {
                        int fx = (int)(result_recognitions[i].box[0].x);
                        int fy = (int)(result_recognitions[i].box[0].y);
                        int fw = (int)((result_recognitions[i].box[3].x - result_recognitions[i].box[0].x));
                        int fh = (int)((result_recognitions[i].box[1].y - result_recognitions[i].box[0].y));
                        fy = (int)(fy * ratio + (UIImageHeight * (1 - ratio))/2.0f + 8);
                        fh = (int)(fh * ratio);

                        DrawRect2D(Color.blue, fx, fy, fw, fh, tex_width, tex_height);
                    }

                }
                else if(output_mode == OutputMode.RecognizedText){

                    for(int i = 0; i < result_recognitions.Count; i++){
                        
                        int fx = (int)(result_recognitions[i].box[0].x);
                        int fy = (int)(result_recognitions[i].box[0].y);
                        fy = (int)(fy * ratio + (UIImageHeight * (1 - ratio))/2.0f + 8);

                        DrawText(Color.white, result_recognitions[i].text, fx, fy, tex_width, tex_height, scale: ratio, text_color: Color.black);
                        //Debug.Log(result_recognitions[i].text);
                    }
                }
            }

            if (label_text != null)
			{
				label_text.text = recognition_time + "ms\n" + ailia_text_recognizer.EnvironmentName();
			}
        }



        private Color32[] ResizeColorArray(Color32[] originalPixels, int originalWidth, int originalHeight, int newWidth, int newHeight)
		{
			Texture2D newTexture = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);

			Texture2D originalTexture = new Texture2D(originalWidth, originalHeight, TextureFormat.RGBA32, false);
			originalTexture.SetPixels32(originalPixels);
			originalTexture.Apply();

            newTexture = GetResized(originalTexture, newWidth, newHeight);

			Color32[] resizedPixels = newTexture.GetPixels32();

			return resizedPixels;
		}


        private Texture2D GetResized(Texture2D texture, int width, int height)
        {
            var rt = RenderTexture.GetTemporary(width, height);
            Graphics.Blit(texture, rt);

            var preRT = RenderTexture.active;
            RenderTexture.active = rt;
            var ret = new Texture2D(width, height);
            ret.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            ret.Apply();
            RenderTexture.active = preRT;

            RenderTexture.ReleaseTemporary(rt);
            return ret;
        }


        void SetUIProperties()
        {
            if (UICanvas == null) return;
            // Set up UI for AiliaDownloader
            var downloaderProgressPanel = UICanvas.transform.Find("DownloaderProgressPanel");
            ailia_download.DownloaderProgressPanel = downloaderProgressPanel.gameObject;
            // Set up lines
            line_panel = UICanvas.transform.Find("LinePanel").gameObject;
            lines = UICanvas.transform.Find("LinePanel/Lines").gameObject;
            line = UICanvas.transform.Find("LinePanel/Lines/Line").gameObject;
            text_panel = UICanvas.transform.Find("TextPanel").gameObject;
            text_base = UICanvas.transform.Find("TextPanel/TextHolder").gameObject;

            raw_image = UICanvas.transform.Find("RawImage").gameObject.GetComponent<RawImage>();
            label_text = UICanvas.transform.Find("LabelText").gameObject.GetComponent<Text>();
            mode_text = UICanvas.transform.Find("ModeLabel").gameObject.GetComponent<Text>();
        }

        void OnApplicationQuit()
        {
            DestroyAiliaDetector();
            ailia_camera.DestroyCamera();
        }

        void OnDestroy()
        {
            DestroyAiliaDetector();
            ailia_camera.DestroyCamera();
        }
    }
}
