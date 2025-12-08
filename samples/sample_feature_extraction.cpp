/*
Copyright 2023 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_efficient_features.h>

#include "sample_common.h"

static std::string keys =
"{ @input-image    |        | input image.                                }"
"{ input-image     |        | input image path(optional, same as @input-image). }"
"{ max-keypoints   |  10000 | maximum number of keypoints.                }"
"{ fast-threshold  |     20 | FAST threshold.                             }"
"{ nonmax-radius   |     15 | radius of non-maximum suppression.          }"
"{ descriptor-type |      0 | descriptor type(0:BAD 1:HashSIFT 2:SphericalBAD 3:SphericalHashSIFT 4:AKAZE 5:SphericalAKAZE 6:ORB 7:SphericalORB). }"
"{ descriptor-bits |    256 | descriptor bits(256 or 512).                }"
"{ compute-async   |        | compute asynchronously.                     }"
"{ no-gui          |        | disable GUI rendering (useful in headless environments). }"
"{ help  h         |        | print help message.                         }";

int main(int argc, char* argv[])
{
        const cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

        const auto collectPositionalArgs = [](int argc, char* argv[]) {
                std::vector<std::string> positional;
                auto consumesValue = [](const std::string& opt) {
                        static const std::set<std::string> withValue = {
                                "--input-image",
                                "--max-keypoints",
                                "--fast-threshold",
                                "--nonmax-radius",
                                "--descriptor-type",
                                "--descriptor-bits",
                        };
                        return withValue.count(opt) > 0;
                };

                for (int i = 1; i < argc; ++i)
                {
                        std::string arg(argv[i]);

                        if (arg.rfind("--", 0) == 0)
                        {
                                const auto eq = arg.find('=');
                                if (eq != std::string::npos)
                                {
                                        continue; // already contains its value
                                }

                                if (consumesValue(arg) && i + 1 < argc)
                                {
                                        ++i; // skip the following value
                                }

                                continue;
                        }

                        if (arg.rfind('-', 0) == 0)
                        {
                                continue; // short options (e.g., -h)
                        }

                        positional.emplace_back(std::move(arg));
                }

                return positional;
        };

        const auto positionalArgs = collectPositionalArgs(argc, argv);
        if (parser.has("input-image") && !positionalArgs.empty())
        {
                std::cerr << "input image was specified both positionally and via --input-image. "
                             "Please provide it only once." << std::endl;
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

        if (positionalArgs.size() > 1)
        {
                std::cerr << "Too many positional arguments were provided. "
                             "Specify exactly one input image (or use --input-image). "
                             "For matching two images, use samples/sample_feature_matching." << std::endl;
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

        // get parameters
        const std::string filename = parser.has("input-image")
                ? parser.get<std::string>("input-image")
                : (positionalArgs.empty() ? std::string{} : positionalArgs.front());
	const int nfeatures = parser.get<int>("max-keypoints");
	const int fastThreshold = parser.get<int>("fast-threshold");
        const int nonmaxRadius = parser.get<int>("nonmax-radius");
        const int descType = sanitizeDescriptorType(parser.get<int>("descriptor-type"));
        const int descBits = normalizeDescriptorBits(descType, parser.get<int>("descriptor-bits"));
        const bool computeAsync = parser.has("compute-async");
        const bool noGui = parser.has("no-gui");

        if (!parser.check())
        {
                parser.printErrors();
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

        if (filename.empty())
        {
                std::cerr << "input image path is required. Specify it as a positional argument or via --input-image." << std::endl;
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

	cv::Mat image = cv::imread(filename);
	if (image.empty())
	{
		std::cerr << "imread failed." << std::endl;
		std::exit(EXIT_FAILURE);
	}

        std::cout << "=== configulations ===" << std::endl;
        std::cout << "image size      : " << image.size() << std::endl;
        std::cout << "descriptor type : " << descriptorTypeName(descType) << std::endl;
	std::cout << "descriptor bits : " << descBits << std::endl;
	std::cout << "max keypoints   : " << nfeatures << std::endl;
	std::cout << "compute async   : " << (computeAsync ? "Yes" : "No") << std::endl;
	std::cout << std::endl;

	cv::Mat gray;
	convertToGray(image, gray);

	// detect keypoints
	auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
	feature->setFastThreshold(fastThreshold);
	feature->setNonmaxRadius(nonmaxRadius);
	feature->setDescriptorType(getDescriptorType(descType, descBits));

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

        if (computeAsync)
        {
                cv::cuda::GpuMat d_gray(gray), d_keypoints, d_descriptors;
		cv::cuda::Stream stream;

		feature->detectAndComputeAsync(d_gray, cv::noArray(), d_keypoints, d_descriptors, false, stream);

		stream.waitForCompletion();
		feature->convert(d_keypoints, keypoints);
		d_descriptors.download(descriptors);
	}
	else
	{
		feature->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        }

        std::cout << keypoints.size() << " keypoints found." << std::endl << std::endl;

        auto makeOutputPath = [](const std::string& path) {
                const auto pos = path.find_last_of("/\\");
                const auto filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
                const auto dot = filename.find_last_of('.');
                const auto stem = (dot == std::string::npos) ? filename : filename.substr(0, dot);
                return stem + "_keypoints.png";
        };

        // draw
        cv::Mat draw;
        drawKeypoints(image, keypoints, draw);

        if (noGui || std::getenv("DISPLAY") == nullptr)
        {
                const auto outputPath = makeOutputPath(filename);
                if (!cv::imwrite(outputPath, draw))
                {
                        std::cerr << "failed to write output image: " << outputPath << std::endl;
                        std::exit(EXIT_FAILURE);
                }

                if (!noGui)
                {
                        std::cerr << "DISPLAY is not set; skipping GUI rendering. Use --no-gui to silence this message." << std::endl;
                }

                std::cout << "output saved to: " << outputPath << std::endl;
        }
        else
        {
                cv::imshow("keypoints", draw);
                cv::waitKey(0);
        }

        return 0;
}
