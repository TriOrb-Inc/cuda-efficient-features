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
"{ @first-image    | <none> | first input image.                          }"
"{ @second-image   | <none> | second input image.                         }"
"{ max-keypoints   |  10000 | maximum number of keypoints.                }"
"{ fast-threshold  |     20 | FAST threshold.                             }"
"{ nonmax-radius   |     15 | radius of non-maximum suppression.          }"
"{ descriptor-type |      0 | descriptor type(0:BAD 1:HashSIFT 2:SphericalBAD 3:SphericalHashSIFT 4:AKAZE 5:SphericalAKAZE 6:ORB 7:SphericalORB). }"
"{ descriptor-bits |    256 | descriptor bits(256 or 512).                }"
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
        if (positionalArgs.size() > 2)
        {
                std::cerr << "Too many positional arguments were provided. "
                             "Specify exactly two input images." << std::endl;
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

        // get parameters
        const std::string filename1 = !positionalArgs.empty() ? positionalArgs.front()
                : parser.get<std::string>("@first-image");
        const std::string filename2 = (positionalArgs.size() == 2) ? positionalArgs.back()
                : parser.get<std::string>("@second-image");
        const int nfeatures = parser.get<int>("max-keypoints");
        const int fastThreshold = parser.get<int>("fast-threshold");
        const int nonmaxRadius = parser.get<int>("nonmax-radius");
        const int descType = sanitizeDescriptorType(parser.get<int>("descriptor-type"));
        const int descBits = normalizeDescriptorBits(descType, parser.get<int>("descriptor-bits"));
        const bool noGui = parser.has("no-gui");

        if (!parser.check())
        {
                parser.printErrors();
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

        if (filename1.empty() || filename2.empty())
        {
                std::cerr << "two input image paths are required. Provide them as positional arguments" << std::endl
                          << "or via @first-image and @second-image." << std::endl;
                parser.printMessage();
                std::exit(EXIT_FAILURE);
        }

        cv::Mat image1 = cv::imread(filename1);
        cv::Mat image2 = cv::imread(filename2);
        if (image1.empty() || image2.empty())
        {
                std::cerr << "imread failed." << std::endl;
                std::exit(EXIT_FAILURE);
        }

        std::cout << "=== configulations ===" << std::endl;
        std::cout << "image size      : " << image1.size() << " and " << image2.size() << std::endl;
        std::cout << "descriptor type : " << descriptorTypeName(descType) << std::endl;
        std::cout << "descriptor bits : " << descBits << std::endl;
        std::cout << "max keypoints   : " << nfeatures << std::endl;
        std::cout << std::endl;

        cv::Mat gray1, gray2;
        convertToGray(image1, gray1);
        convertToGray(image2, gray2);

        // extract features
        std::cout << "=== extract features ===" << std::endl;
        auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        feature->setFastThreshold(fastThreshold);
        feature->setNonmaxRadius(nonmaxRadius);
        feature->setDescriptorType(getDescriptorType(descType, descBits));

        const auto needsClassId = descType == AKAZE || descType == SphericalAKAZE;
        const auto assignClassIdIfNeeded = [&](std::vector<cv::KeyPoint>& keypoints) {
                if (!needsClassId)
                        return;

                for (auto& kp : keypoints)
                {
                        kp.class_id = 0;
                }
        };

        feature->detect(gray1, keypoints1, cv::noArray());
        assignClassIdIfNeeded(keypoints1);
        feature->compute(gray1, keypoints1, descriptors1);

        feature->detect(gray2, keypoints2, cv::noArray());
        assignClassIdIfNeeded(keypoints2);
        feature->compute(gray2, keypoints2, descriptors2);

        std::cout << "number of keypoins: " << keypoints1.size() << " " << keypoints2.size() << std::endl;

        // match features
        std::cout << "=== match features ===" << std::endl;
        auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher->match(descriptors1, descriptors2, matches);

        std::cout << "number of matches: " << matches.size() << std::endl;

        // draw
        cv::Mat draw;
        drawMatches(image1, keypoints1, image2, keypoints2, matches, draw);

        const auto makeOutputPath = [](const std::string& lhs, const std::string& rhs) {
                const auto basename = [](const std::string& path) {
                        const auto pos = path.find_last_of("/\\");
                        return (pos == std::string::npos) ? path : path.substr(pos + 1);
                };

                const auto stem = [](const std::string& filename) {
                        const auto dot = filename.find_last_of('.');
                        return (dot == std::string::npos) ? filename : filename.substr(0, dot);
                };

                const auto lhsStem = stem(basename(lhs));
                const auto rhsStem = stem(basename(rhs));
                return lhsStem + "_vs_" + rhsStem + "_matches.png";
        };

        if (noGui || std::getenv("DISPLAY") == nullptr)
        {
                const auto outputPath = makeOutputPath(filename1, filename2);
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
                cv::imshow("image", draw);
                cv::waitKey(0);
        }

        return 0;
}
