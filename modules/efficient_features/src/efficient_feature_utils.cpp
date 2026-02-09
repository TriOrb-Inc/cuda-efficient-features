/*
Copyright 2024 TriOrb Inc.

The major design pattern of this plugin was abstracted
from Fixstars Corporation, which is subject to the same license.
Here is the original copyright notice:

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

#include "efficient_descriptors.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{

void detectAndComputePerChannel(const Ptr<Feature2D>& feature, InputArray image, InputArray mask,
        std::vector<std::vector<KeyPoint>>& keypoints, std::vector<Mat>& descriptors,
        bool useProvidedKeypoints)
{
        CV_Assert(feature);
        CV_Assert(image.depth() == CV_8U);

        if (!mask.empty())
        {
                CV_Assert(mask.type() == CV_8U);
                CV_Assert(mask.channels() == 1);
        }

        const Mat imageMat = image.getMat();
        const int channels = imageMat.channels();
        CV_Assert(channels >= 1);

        std::vector<Mat> planes;
        if (channels == 1)
        {
                planes.emplace_back(imageMat);
        }
        else
        {
                split(imageMat, planes);
        }

        if (useProvidedKeypoints)
                CV_Assert(static_cast<int>(keypoints.size()) == channels);

        keypoints.resize(channels);
        descriptors.resize(channels);

        for (int c = 0; c < channels; ++c)
                feature->detectAndCompute(planes[c], mask, keypoints[c], descriptors[c], useProvidedKeypoints);
}

} // namespace cv
