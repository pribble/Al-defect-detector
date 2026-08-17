#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <sstream>
#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <signal.h>
#include <arm_neon.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "paddle_api.h"  // NOLINT
#include "core/tensor.h"
#include "intelfpga.h"
#include "httplib.h"

using namespace paddle::lite_api;  // NOLINT
using namespace httplib;

#define input_shape 300  // 必须是2的整数倍

struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
};

// Object for storing all preprocessed data
struct ImageBlob {
  float mean_;
  float scale_;
};

std::vector<std::string> ReadLine(const std::string &path) {
  std::ifstream file(path);
  std::vector<std::string> line_vec;
  std::string line;
  if (file){
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      line_vec.push_back(line);
    }
  }
  return line_vec;
}

std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) res.push_back(item);
  }
  return res;
}

std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
  auto config = ReadLine(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], ' ');
    if (res.size() >= 2) dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "=======PaddleDetection lite demo config======" << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "===End of PaddleDetection lite demo config===" << std::endl;
}

class Detector {
public:
  explicit Detector(const std::string& config_path) {
    config_ = LoadConfigTxt(config_path);
    PrintConfig(config_);

    std::string label_path = config_.at("label_path");
    class_names_ = ReadLine(label_path);

    // label_list 每行格式: <class_name> [threshold]
    // 第二列缺失时默认 0.45 (向后兼容)
    class_thresholds_.resize(class_names_.size(), 0.45f);
    for (size_t i = 0; i < class_names_.size(); i++) {
      auto parts = split(class_names_[i], ' ');
      if (parts.size() >= 2) {
        class_names_[i] = parts[0];
        class_thresholds_[i] = std::stof(parts[1]);
      }
    }

    loadModel();
    predictor_->InitInputs();

    // Pre-fill input[0] — im_shape:  always [300, 300]
    {
      float* d = predictor_->GetInput(0)->mutable_data<float>();
      d[0] = d[1] = input_shape;
    }
    // scale 输入恒为 {1, 1}
    {
      float* scale_data = predictor_->GetInput(2)->mutable_data<float>();
      scale_data[0] = scale_data[1] = 1.f;
    }

    init_ImageData();

    // 启动 warmup：构造完成立即跑一次空图推理
    cv::Mat warmup_img(input_shape, input_shape, CV_8UC1);
    Detect(warmup_img);
  }

  ~Detector() {
    fpga_release();
  }

  std::string GetClassName(int class_id) const {
    if (class_id >= 0 && class_id < (int)class_names_.size()) {
      return class_names_[class_id];
    }
    return "unknown";
  }

  std::vector<Object> Detect(const cv::Mat& frame) {
    if (frame.empty()) {
      std::cerr << "[WARN] processFrame received empty frame\n";
      return {};
    }

    // 本检测器只适配灰度图：非 1 通道输入直接拒绝，不自动转换。
    if (frame.channels() != 1) {
      std::cerr << "[WARN] Detect only accepts grayscale (1-channel) input, got "
                << frame.channels() << " channels\n";
      return {};
    }

    // input[1] — image data (changes every frame).
    // Shape [1, 3, 300, 300] was pre-set by InitInputs.
    // input[2] (scale) 已在构造时写死为 {1, 1}，无需每帧更新。
    float* data1 = predictor_->GetInput(1)->mutable_data<float>();
    preprocessImg(frame, data1);

    predictor_->Run();

    const auto* output_tensor = predictor_->GetOutput();
    // outdata : [class_id, confidence, x_min, y_min, x_max, y_max]
    const float* outdata = output_tensor->data<float>();
    int64_t count = output_tensor->dims().production();
    int num_detections = static_cast<int>(count / 6);

    std::vector<Object> objects;
    for (int iw = 0; iw < num_detections; iw++) {
      int class_id = static_cast<int>(outdata[0]);
      float threshold = (class_id >= 0 && class_id < (int)class_thresholds_.size())
                          ? class_thresholds_[class_id] : 0.45f;
      if (threshold < outdata[1] && outdata[1] <= 1 && outdata[2] < outdata[4] && outdata[3] < outdata[5]) {
        Object obj;
        int x = static_cast<int>(outdata[2]);
        int y = static_cast<int>(outdata[3]);
        int w = static_cast<int>(outdata[4] - outdata[2]);
        int h = static_cast<int>(outdata[5] - outdata[3]);
        cv::Rect rec_clip = cv::Rect(x, y, w, h) & cv::Rect(0, 0, frame.cols, frame.rows);
        obj.class_id = class_id;
        obj.prob = outdata[1];
        obj.rec = rec_clip;
        objects.push_back(obj);
      }
      outdata += 6;
    }
    return objects;
  }

  void visualize_result(std::vector<Object> objects, cv::Mat& image) {
    int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 1.f;
    int thickness = 1;
    const cv::Scalar kBoxColor(255);   // 白色框
    const cv::Scalar kTextColor(255);  // 白色文字
    for (auto obj: objects){
      auto rec_clip = obj.rec;
      std::string class_name = GetClassName(obj.class_id);
      std::cout << "image size: " << image.cols << ", " << image.rows
                << ", detect object: " << class_name << ", score: " << obj.prob
                << ", location: x=" << rec_clip.x << ", y=" << rec_clip.y
                << ", width=" << rec_clip.width << ", height=" << rec_clip.height
                << std::endl;
      cv::rectangle(image, rec_clip, kBoxColor, 1, cv::LINE_AA);
      std::string str_prob = std::to_string(obj.prob);
      std::string text = class_name + ": " +
                          str_prob.substr(0, str_prob.find(".") + 4);
      cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
      float new_font_scale = rec_clip.width * 0.5f * font_scale / text_size.width;
      text_size = cv::getTextSize(text, font_face, new_font_scale, thickness, nullptr);
      cv::Point origin;
      origin.x = rec_clip.x + 3;
      origin.y = rec_clip.y + text_size.height + 3;
      cv::putText(image,
                  text,
                  origin,
                  font_face,
                  new_font_scale,
                  kTextColor,
                  thickness,
                  cv::LINE_AA);
    }
  }

private:
  std::map<std::string, std::string> config_;
  std::shared_ptr<PaddlePredictor> predictor_;
  std::vector<std::string> class_names_;
  std::vector<float> class_thresholds_;
  ImageBlob img_data;
  cv::Mat img_float;

  void loadModel() {
    std::string model_file = config_.at("model_file");

    predictor_ = std::make_shared<PaddlePredictor>(model_file);
    if (!predictor_) {
      std::cerr << "[ERROR] Failed to create predictor\n";
      exit(1);
    }
  }

  void init_ImageData() {
    img_data.mean_ = std::stof(config_.at("mean"));
    img_data.scale_ = std::stof(config_.at("std"));
  }

  // 灰度图 (CV_8UC1) → 3-channel NCHW tensor
  void preprocessImg(const cv::Mat& img, float* data) {
    img.convertTo(img_float, CV_32FC1, 1.0);
    const float* din = reinterpret_cast<const float*>(img_float.data);

    const int size = input_shape * input_shape;
    float* dout_c0 = data;
    float* dout_c1 = data + size;
    float* dout_c2 = data + size * 2;

    float32x4_t vmean = vdupq_n_f32(img_data.mean_);
    float32x4_t vscale = vdupq_n_f32(1.f / img_data.scale_);

    for (int i = 0; i < size; i += 4) {
      float32x4_t vin = vld1q_f32(din + i);
      float32x4_t vscaled = vmulq_f32(vsubq_f32(vin, vmean), vscale);
      vst1q_f32(dout_c0 + i, vscaled);
      vst1q_f32(dout_c1 + i, vscaled);
      vst1q_f32(dout_c2 + i, vscaled);
    }
  }
};

void detect_image_file(Detector& detector, std::string input_path){
  std::vector<std::string> img_paths;
  struct stat st;
  if (stat(input_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
    DIR* dp = opendir(input_path.c_str());
    dirent* entry;
    while (entry = readdir(dp)){
      if (entry->d_name[0] == '.') continue;
      img_paths.emplace_back(input_path + '/' + entry->d_name);
    }
    closedir(dp);
  }
  else img_paths.push_back(input_path);

  if (stat("./result", &st) != 0) {
      mkdir("./result", 0755);
  }

  std::cout<< "\n#######################################\n" <<std::endl;
  for (const auto& img_path: img_paths){
    std::cout<< img_path <<std::endl;

    cv::Mat img = imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "[ERROR] Failed to read image: " << img_path << "\n";
      continue;
    }

    auto start = std::chrono::steady_clock::now();

    // Resize to model input size
    cv::resize(img, img, cv::Size(input_shape, input_shape), 0, 0, cv::INTER_CUBIC);

    auto objects = detector.Detect(img);
    detector.visualize_result(objects, img);

    auto end = std::chrono::steady_clock::now();

    std::cout<< "total time spent(ms): "
             << std::chrono::duration<float, std::milli>(end - start).count()
             <<std::endl;

    std::string img_name = img_path.substr(img_path.find_last_of("/\\") + 1);
    std::string result_name = "./result/" + img_name.substr(0, img_name.find_last_of('.')) + "_result.jpg";
    cv::imwrite(result_name, img);
    std::cout << "Result saved to " << result_name << std::endl;
    std::cout<< "\n#######################################\n" <<std::endl;
  }
}

void detect_camera_frame(Detector& detector){
  Server svr;
  svr.new_task_queue = [] { return new ThreadPool(1); };
  svr.Post("/predict", [&detector](const Request &req, Response &res) {
    auto image_file = req.get_file_value("image_file");
    // Pi sends raw 300×300 grayscale pixels (no JPEG encode/decode, no cvtColor)
    if (image_file.content.size() != input_shape * input_shape) {
      std::cerr << "[ERROR] Invalid image data size: " << image_file.content.size() << "\n";
      res.set_content("{\"len\":0,\"action\":\"OK\",\"result\":[]}", "application/json");
      return;
    }
    cv::Mat img_decode(input_shape, input_shape, CV_8UC1, (void*)image_file.content.data());

    if (img_decode.empty()) {
      std::cerr << "[ERROR] Failed to decode image\n";
      res.set_content("{\"len\":0,\"result\":[]}", "application/json");
      return;
    }

    auto start = std::chrono::steady_clock::now();
    auto objects = detector.Detect(img_decode);
    auto end = std::chrono::steady_clock::now();
    float prediction_time = std::chrono::duration<float, std::milli>(end - start).count();

    // Build JSON response matching HaoYao GrabImage expectations
    std::ostringstream json;
    bool has_defect = objects.size() > 0;
    json << "{\"len\":" << objects.size()
         << ",\"action\":\"" << (has_defect ? "NG" : "OK") << "\""
         << ",\"result\":[";
    for (size_t i = 0; i < objects.size(); i++) {
      if (i > 0) json << ",";
      auto& obj = objects[i];
      json << "{"
           << "\"class_name\":\"" << detector.GetClassName(obj.class_id) << "\","
           << "\"loc\":[" << obj.rec.x << "," << obj.rec.y << ","
           << (obj.rec.x + obj.rec.width) << "," << (obj.rec.y + obj.rec.height) << "],"
           << "\"score\":" << obj.prob << ","
           << "\"prediction_time\":" << prediction_time
           << "}";
    }
    json << "]}";

    res.set_content(json.str(), "application/json");
  });
  svr.listen("0.0.0.0", 8080);
}

void int_handler(int sig){
  fflush(stdout);
  fpga_release();
  std::cerr << "SIGINT: stopping the predictor\n";
  exit(-1);
}

int main(int argc, char** argv) {
  if (argc < 2){
    std::cerr << "\n[ERROR] Missing required configuration path argument\n";
    std::cerr << "Usage: " << argv[0] << " <config_path> [image_path|image_dir]\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  <config_path>      Required: Path to configuration file\n";
    std::cerr << "  [image_path|image_dir] Optional: Single image file or directory path\n";
    return 1;
  }
  std::string config_path = argv[1];

  signal(SIGINT, int_handler);

  Detector detector(config_path);

  if (argc == 2) detect_camera_frame(detector);
  else{
    std::string input_path = argv[2];
    detect_image_file(detector, input_path);
  }

  return 0;
}
