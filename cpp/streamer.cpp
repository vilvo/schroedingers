#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/hal/interface.h> // CV_8UC3
#include <opencv2/highgui.hpp>          // imshow
#include <opencv2/imgproc.hpp>          // putText
#include <opencv2/core/types_c.h>       // cvScalar
#include <string>

// protocol buffer message
#include "image.pb.h"

const std::string DEBUG_WINDOW_NAME("debug window");

cv::Mat overlay_text(const std::string text, const cv::Mat& image) {
  cv::Mat overlayed(image);
  cv::putText(overlayed, text, cvPoint(30,30), // a little to the center from top left
	      cv::FONT_HERSHEY_COMPLEX_SMALL,
	      0.8, cvScalar(200,200,250),
	      1, CV_AA);
  return overlayed;
}

void visualdebug(const cv::Mat& image, const auto& framecount) {
  auto debugframe = overlay_text(std::to_string(framecount), image);
  cv::namedWindow(DEBUG_WINDOW_NAME);
  cv::imshow(DEBUG_WINDOW_NAME, debugframe);
  cv::waitKey(1);
}

rs2::config get_pipeline_config() {
  rs2::config cfg;
  // set colors to support conversion to cv::Mat with CV_8UC3
  // resize to the smallest supported by camera
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  return cfg;
}

//
bool encode_jpeg(const cv::Mat input, std::vector<uchar> & output) {
  const std::string JPEG(".jpeg");
  std::vector<int> param(2);
  param[0] = cv::IMWRITE_JPEG_QUALITY;
  param[1] = 70; // jpeg quality between 0-100
  return cv::imencode(JPEG, input, output, param);
}

int main(int argc, char * argv[]) try
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  rs2::pipeline pipe;
  pipe.start(get_pipeline_config());

  auto debug = 1, framecount = 0;

  while (1) {
    // wait pipeline to have new data available
    rs2::frameset data = pipe.wait_for_frames();
    framecount += 1;
    // filter only colorframe from the dataset (ditch depth frame)
    auto colorframe = data.get_color_frame();
    // convert colorframe to opencv mat
    cv::Mat image(cv::Size(colorframe.get_width(),
			   colorframe.get_height()),
		  CV_8UC3,
		  (void*)colorframe.get_data(),
		  cv::Mat::AUTO_STEP);

    if(debug) {
      visualdebug(image, framecount);
    }

    //TODO: resize mat, normalize for NN input layer?

    std::vector<uchar> jpeg;
    // encode to jpeg
    if (!encode_jpeg(image, jpeg)) {
      std::cerr << "encoding to jpeg failed" << std::endl;
      continue;
    }

    // map jpeg to protobuf image
    schroedingers::image pb_image;
    pb_image.set_width(colorframe.get_width());
    pb_image.set_height(colorframe.get_height());
    pb_image.set_data(std::string(jpeg.begin(), jpeg.end()));

    // TODO: serialize/stream to py

  }

  return EXIT_SUCCESS;
 } // main
 // boilerplate exception handling from realsense examples
 catch (const rs2::error & e) {
     std::cerr << "realsense error calling "
	       << e.get_failed_function()
	       << "(" << e.get_failed_args() << "):\n    "
	       << e.what() << std::endl;
     return EXIT_FAILURE;
 }
 catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     return EXIT_FAILURE;
 }
