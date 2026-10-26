# Models and environments

Model weights are not distributed in this repository. Users must accept and
follow the original model licenses.

The paper used more than one serving environment because the evaluated model
families did not all run under the same vLLM release. Direct dependency
versions are recorded in `configs/environments/`.

Before public release, add for each model:

- official model identifier and exact revision;
- license or terms link;
- serving environment;
- image/video preprocessing settings;
- context and generation limits; and
- tested GPU type and minimum memory.

Do not include personal conda prefixes, cache directories, hostnames, or
cluster-specific module commands.
