Testing

This document explains how to write unit tests, E2E tests, and nightly tests to verify your feature implementation.

Set up a test environment

The fastest way to set up a test environment is to use the main branch's container image:

=== "Local (CPU)"

    You can run the unit tests on CPUs with the following steps:

    

=== "Single card"

    

    After starting the container, you should install the required packages:

    

=== "Multi cards"

    

    After starting the container, you should install the required packages:

    

Running tests

Unit tests

There are several principles to follow when writing unit tests:

- The test file path should be consistent with the source file and start with the test_ prefix, such as: vllm_ascend/worker/worker.py --> tests/ut/worker/test_worker.py
- The vLLM Ascend test uses unittest framework. See the Python unittest documentation to understand how to write unit tests.
- All unit tests can be run on CPUs, so you must mock the device-related functions on the host.
- Example: tests/ut/test_ascend_config.py.
- You can run the unit tests using pytest:

=== "Local (CPU)"

    

=== "Single-card"

    

=== "Multi-card"

    

E2E test

Although vllm-ascend CI provides E2E tests on Ascend CI (for example,
schedule_nightly_test_a2.yaml, schedule_nightly_test_a3.yaml, pr_test.yaml), you can run them locally.

PR-triggered E2E test

You can run tests with pytest as well. Typical examples:

=== "Local (CPU)"

    You can't run the E2E test on CPUs.

=== "Single-card"

    

=== "Multi-card"

    

This will reproduce the E2E test behavior.

Nightly-triggered E2E test

You can run tests with pytest as well. Typical examples:

=== "Local (CPU)"

    You can't run the E2E test on CPUs.

=== "Single-card"

    

=== "Multi-card"

    

For running nightly single-node model test cases locally, refer to the following example.



For running nightly multi-node model test cases locally, refer to the Running Locally section in Multi Node Test.

E2E test examples

- Offline test example: tests/e2e/pull_request/one_card/test_camem.py
- Online test example: tests/e2e/pull_request/two_card/aclgraph/test_single_request_aclgraph.py
- Correctness test example: tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py

The CI resource is limited, and you might need to reduce the number of layers of a model. Below is an example of how to generate a reduced layer model:

1. Fork the original model repo in modelscope. All the files in the repo except for weights are required.
2. Set num_hidden_layers to the expected number of layers, e.g., {"num_hidden_layers": 2,}
3. Copy the following python script as generate_random_weight.py. Set the relevant parameters MODEL_LOCAL_PATH, DIST_DTYPE and DIST_MODEL_PATH as needed:

    

Run doctest

vllm-ascend provides a vllm-ascend/tests/e2e/run_doctests.sh command to run all doctests in the doc files.
The doctest is a good way to make sure docs stay current and examples remain executable, which can be run locally as follows:



This will reproduce the same environment as the CI. See labeled_doctest.yaml.

Run docs link check

You can validate external links in the Sphinx docs locally with:



To check links in a specific Markdown file, pass the file to sphinx-build.
For example, to check only docs/source/user_guide/release_notes.md:



The detailed report will be written to:

- docs/_build/linkcheck/output.txt
- docs/_build/linkcheck/output.json
