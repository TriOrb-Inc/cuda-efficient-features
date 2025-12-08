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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_efficient_features.h>
#include <cuda_efficient_descriptors.h>

#include "sample_common.h"

static std::string keys =
"{ @first-image    | <none> | first input image.                          }"
"{ @second-image   | <none> | second input image.                         }"
"{ max-keypoints   |  10000 | maximum number of keypoints.                }"
"{ fast-threshold  |     20 | FAST threshold.                             }"
"{ nonmax-radius   |     15 | radius of non-maximum suppression.          }"
"{ descriptor-type |      0 | descriptor type(0:BAD 1:HashSIFT 2:SphericalBAD 3:SphericalHashSIFT 4:AKAZE 5:SphericalAKAZE 6:ORB 7:SphericalORB). }"
"{ descriptor-bits |    256 | descriptor bits(256 or 512).                }"
"{ fx             |      0 | horizontal focal length for spherical descriptors.     }"
"{ fy             |      0 | vertical focal length for spherical descriptors.       }"
"{ cx             |      0 | principal point x for spherical descriptors.           }"
"{ cy             |      0 | principal point y for spherical descriptors.           }"
"{ k1             |      0 | radial distortion k1 for spherical descriptors.        }"
"{ k2             |      0 | radial distortion k2 for spherical descriptors.        }"
"{ k3             |      0 | radial distortion k3 for spherical descriptors.        }"
"{ k4             |      0 | radial distortion k4 for spherical descriptors.        }"
"{ no-gui          |        | disable GUI rendering (useful in headless environments). }"
"{ help  h         |        | print help message.                         }";

struct Dataset
{
        std::filesystem::path left;
        std::filesystem::path right;
        std::string name;
};

static std::filesystem::path findInputRoot(const std::filesystem::path& inputPath)
{
        auto current = inputPath;
        while (!current.empty())
        {
                if (current.filename() == "input")
                        return current;

                current = current.parent_path();
        }

        return std::filesystem::path{};
}

static std::vector<Dataset> resolveDatasets(const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
        std::vector<Dataset> datasets;

        const auto lhsRoot = findInputRoot(lhs);
        const auto rhsRoot = findInputRoot(rhs);

        if (!lhsRoot.empty() && lhsRoot == rhsRoot)
        {
                for (const auto& dir : std::filesystem::directory_iterator(lhsRoot))
                {
                        if (!dir.is_directory())
                                continue;

                        const auto left = dir.path() / "left.jpg";
                        const auto right = dir.path() / "right.jpg";
                        if (std::filesystem::exists(left) && std::filesystem::exists(right))
                        {
                                datasets.push_back({left, right, dir.path().filename().string()});
                        }
                }
        }

        if (datasets.empty())
        {
            const auto leftParent = lhs.parent_path();
            datasets.push_back({lhs, rhs, leftParent.filename().string()});
        }

        std::sort(datasets.begin(), datasets.end(), [](const Dataset& a, const Dataset& b) {
                return a.name < b.name;
        });

        return datasets;
}

static std::filesystem::path findResultsPath(const std::filesystem::path& start)
{
        auto current = start;
        std::error_code ec;

        while (true)
        {
                        const auto candidate = current / "RESULTS.md";
                        if (std::filesystem::exists(candidate, ec))
                                return candidate;

                        if (!current.has_parent_path() || current.parent_path() == current)
                                break;

                        current = current.parent_path();
        }

        return start / "RESULTS.md";
}

static void updateResults(
        const std::filesystem::path& resultsPath,
        const std::string& descriptorLabel,
        const std::string& datasetName,
        const std::filesystem::path& relativeOutputPath)
{
        const std::string heading = "## ./images/input/" + datasetName + "/left.jpg vs right.jpg";

        std::vector<std::string> lines;
        {
                std::ifstream ifs(resultsPath);
                if (!ifs)
                {
                        lines.emplace_back("# Matches");
                        lines.emplace_back("");
                }
                else
                {
                        std::string line;
                        while (std::getline(ifs, line))
                        {
                                lines.emplace_back(std::move(line));
                        }
                }
        }

        const auto ensureTrailingBlank = [&]() {
                if (!lines.empty() && !lines.back().empty())
                        lines.emplace_back("");
        };

        const auto findSection = [&]() {
                for (size_t i = 0; i < lines.size(); ++i)
                {
                        if (lines[i] == heading)
                                return static_cast<long long>(i);
                }
                return -1LL;
        };

        auto sectionIndex = findSection();
        if (sectionIndex < 0)
        {
                ensureTrailingBlank();
                sectionIndex = static_cast<long long>(lines.size());
                lines.push_back(heading);
                lines.push_back("|Feature Type|Image|");
                lines.push_back("|:--:|:--:|");
        }

        const auto sectionEnd = [&]() {
                for (size_t i = static_cast<size_t>(sectionIndex + 1); i < lines.size(); ++i)
                {
                        if (lines[i].rfind("## ", 0) == 0)
                                return static_cast<long long>(i);
                }
                return static_cast<long long>(lines.size());
        }();

        const std::string row = "|" + descriptorLabel + "|![" + descriptorLabel + "](" + relativeOutputPath.generic_string() + ")|";

        bool exists = false;
        for (size_t i = static_cast<size_t>(sectionIndex + 1); i < static_cast<size_t>(sectionEnd); ++i)
        {
                if (lines[i] == row)
                {
                        exists = true;
                        break;
                }
        }

        if (!exists)
        {
                lines.insert(lines.begin() + sectionEnd, row);
        }

        std::ofstream ofs(resultsPath, std::ios::trunc);
        for (size_t i = 0; i < lines.size(); ++i)
        {
                ofs << lines[i];
                if (i + 1 < lines.size())
                        ofs << '\n';
        }
}

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

        const cv::cuda::SphericalLensParams lensParams = {
                parser.get<float>("fx"), parser.get<float>("fy"), parser.get<float>("cx"), parser.get<float>("cy"),
                parser.get<float>("k1"), parser.get<float>("k2"), parser.get<float>("k3"), parser.get<float>("k4"),
        };

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

        const auto datasets = resolveDatasets(std::filesystem::path(filename1), std::filesystem::path(filename2));

        const auto exePath = std::filesystem::weakly_canonical(std::filesystem::path(argv[0]));
        const auto resultsPath = findResultsPath(exePath.parent_path());
        const auto repoRoot = resultsPath.parent_path();
        const auto resolveOutputDir = [&]() {
                auto current = exePath.parent_path();
                std::error_code ec;

                while (true)
                {
                        const auto candidate = current / "images" / "output";
                        if (std::filesystem::exists(candidate, ec) && std::filesystem::is_directory(candidate, ec))
                        {
                                return candidate;
                        }

                        if (!current.has_parent_path() || current.parent_path() == current)
                        {
                                break;
                        }

                        current = current.parent_path();
                }

                return exePath.parent_path() / "images" / "output";
        }();

        const auto descriptorLabel = std::string(descriptorTypeName(descType)) + std::to_string(descBits);

        const auto makeOutputPath = [&](const std::filesystem::path& datasetDir, const std::string& lhs, const std::string& rhs) {
                const auto stem = [](const std::string& filename) {
                        const auto dot = filename.find_last_of('.');
                        return (dot == std::string::npos) ? filename : filename.substr(0, dot);
                };

                const auto lhsStem = stem(lhs);
                const auto rhsStem = stem(rhs);
                const auto prefix = descriptorLabel + "_";
                return resolveOutputDir / datasetDir / (prefix + lhsStem + "_vs_" + rhsStem + "_matches.jpg");
        };

        auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
        feature->setFastThreshold(fastThreshold);
        feature->setNonmaxRadius(nonmaxRadius);
        feature->setDescriptorType(getDescriptorType(descType, descBits));
        feature->setSphericalLensParams(lensParams);

        const auto needsClassId = descType == AKAZE || descType == SphericalAKAZE;
        const auto assignClassIdIfNeeded = [&](std::vector<cv::KeyPoint>& keypoints) {
                if (!needsClassId)
                        return;

                for (auto& kp : keypoints)
                {
                        kp.class_id = 0;
                }
        };

        std::error_code ec;
        std::filesystem::create_directories(resolveOutputDir, ec);
        if (ec)
        {
                std::cerr << "failed to prepare output directory: " << resolveOutputDir << " (" << ec.message() << ")" << std::endl;
                std::exit(EXIT_FAILURE);
        }

        for (const auto& dataset : datasets)
        {
                const auto image1 = cv::imread(dataset.left.string());
                const auto image2 = cv::imread(dataset.right.string());
                if (image1.empty() || image2.empty())
                {
                        std::cerr << "imread failed for dataset: " << dataset.name << std::endl;
                        std::exit(EXIT_FAILURE);
                }

                std::cout << "=== configulations ===" << std::endl;
                std::cout << "dataset         : " << dataset.name << std::endl;
                std::cout << "image size      : " << image1.size() << " and " << image2.size() << std::endl;
                std::cout << "descriptor type : " << descriptorTypeName(descType) << std::endl;
                std::cout << "descriptor bits : " << descBits << std::endl;
                std::cout << "max keypoints   : " << nfeatures << std::endl;
                std::cout << std::endl;

                cv::Mat gray1, gray2;
                convertToGray(image1, gray1);
                convertToGray(image2, gray2);

                std::vector<cv::KeyPoint> keypoints1, keypoints2;
                cv::Mat descriptors1, descriptors2;

                feature->detect(gray1, keypoints1, cv::noArray());
                assignClassIdIfNeeded(keypoints1);
                feature->compute(gray1, keypoints1, descriptors1);

                feature->detect(gray2, keypoints2, cv::noArray());
                assignClassIdIfNeeded(keypoints2);
                feature->compute(gray2, keypoints2, descriptors2);

                std::cout << "number of keypoins: " << keypoints1.size() << " " << keypoints2.size() << std::endl;

                // match features
                std::cout << "=== match features ===" << std::endl;
                auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);

                std::vector<std::vector<cv::DMatch>> knnMatches;
                matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

                // ratio test to reject ambiguous correspondences
                // A slightly looser ratio threshold gives RANSAC more candidates to work with.
                constexpr float ratioThreshold = 0.80f;
                std::vector<cv::DMatch> goodMatches;
                goodMatches.reserve(knnMatches.size());
                for (const auto& m : knnMatches)
                {
                        if (m.size() < 2)
                                continue;

                        if (m[0].distance < ratioThreshold * m[1].distance)
                        {
                                goodMatches.emplace_back(m[0]);
                        }
                }

                // geometric verification with RANSAC
                std::vector<cv::DMatch> matches;
                if (goodMatches.size() >= 4)
                {
                        std::vector<cv::Point2f> pts1, pts2;
                        pts1.reserve(goodMatches.size());
                        pts2.reserve(goodMatches.size());

                        for (const auto& match : goodMatches)
                        {
                                pts1.emplace_back(keypoints1[match.queryIdx].pt);
                                pts2.emplace_back(keypoints2[match.trainIdx].pt);
                        }

                        const auto countGridCoverage = [&](const std::vector<unsigned char>& mask) {
                                constexpr int gridCols = 4;
                                constexpr int gridRows = 4;
                                std::array<bool, gridCols * gridRows> occupied{};

                                for (size_t i = 0; i < mask.size(); ++i)
                                {
                                        if (!mask[i])
                                                continue;

                                        const auto& pt = pts1[i];
                                        const int col = std::clamp(static_cast<int>(pt.x * gridCols / image1.cols), 0, gridCols - 1);
                                        const int row = std::clamp(static_cast<int>(pt.y * gridRows / image1.rows), 0, gridRows - 1);
                                        occupied[row * gridCols + col] = true;
                                }

                                return static_cast<size_t>(std::count(occupied.begin(), occupied.end(), true));
                        };

                        auto runRansac = [&](double reprojectionThreshold) {
                                std::vector<unsigned char> inliersMask;
                                const cv::Mat H = cv::findHomography(
                                        pts1, pts2, cv::RANSAC, reprojectionThreshold, inliersMask, 5000, 0.999);
                                return std::make_pair(H, std::move(inliersMask));
                        };

                        std::vector<unsigned char> bestMask;
                        double bestScore = -1.0;

                        for (const double reproj : {3.0, 5.0, 8.0})
                        {
                                auto [H, inliersMask] = runRansac(reproj);
                                if (H.empty())
                                        continue;

                                const auto inliers = std::count(inliersMask.begin(), inliersMask.end(), 1);
                                const auto coverage = countGridCoverage(inliersMask);
                                const double score = static_cast<double>(inliers) + 1.0 * static_cast<double>(coverage);
                                if (score > bestScore)
                                {
                                        bestScore = score;
                                        bestMask = std::move(inliersMask);
                                }
                        }

                        if (!bestMask.empty())
                        {
                                matches.reserve(goodMatches.size());
                                for (size_t i = 0; i < goodMatches.size(); ++i)
                                {
                                        if (bestMask[i])
                                        {
                                                matches.emplace_back(goodMatches[i]);
                                        }
                                }
                        }
                }

                if (matches.empty())
                {
                        matches = std::move(goodMatches);
                }

                std::cout << "number of matches: " << matches.size() << std::endl;

                // draw
                cv::Mat draw;
                drawMatches(image1, keypoints1, image2, keypoints2, matches, draw);

                const auto putOverlayText = [&](cv::Mat& img) {
                        const int margin = 12;
                        const double fontScale = 0.7;
                        const int thickness = 2;
                        const cv::Scalar textColor(255, 255, 255);
                        const cv::Scalar shadowColor(0, 0, 0);

                        const auto drawTextLine = [&](const std::string& text, int line) {
                                const cv::Point org(margin, margin + (line + 1) * 24);
                                cv::putText(img, text, org + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness, cv::LINE_AA);
                                cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
                        };

                        drawTextLine("Keypoints: " + std::to_string(keypoints1.size()) + " / " + std::to_string(keypoints2.size()), 0);
                        drawTextLine("Inliers: " + std::to_string(matches.size()), 1);
                };

                putOverlayText(draw);

                const auto datasetOutputDir = resolveOutputDir / dataset.name;
                std::filesystem::create_directories(datasetOutputDir, ec);
                if (ec)
                {
                        std::cerr << "failed to prepare dataset output directory: " << datasetOutputDir << " (" << ec.message() << ")" << std::endl;
                        std::exit(EXIT_FAILURE);
                }

                if (noGui || std::getenv("DISPLAY") == nullptr)
                {
                        const auto outputPath = makeOutputPath(dataset.name, dataset.left.filename().string(), dataset.right.filename().string());
                        if (!cv::imwrite(outputPath.string(), draw))
                        {
                                std::cerr << "failed to write output image: " << outputPath << std::endl;
                                std::exit(EXIT_FAILURE);
                        }

                        if (!noGui)
                        {
                                std::cerr << "DISPLAY is not set; skipping GUI rendering. Use --no-gui to silence this message." << std::endl;
                        }

                        const auto relativePath = std::filesystem::relative(outputPath, repoRoot);
                        std::cout << "output saved to: " << relativePath << std::endl;

                        updateResults(resultsPath, descriptorLabel, dataset.name, std::filesystem::relative(outputPath, repoRoot));
                }
                else
                {
                        cv::imshow("image", draw);
                        cv::waitKey(0);
                }
        }

        return 0;
}
