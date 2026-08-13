# Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code Author: Jens Kreber <jens.kreber@uni-a.de>


class NotADict(dict):
    def __getattribute__(self, name: str):
        if name in self:
            return self[name]
        return super().__getattribute__(name)


config = NotADict()
