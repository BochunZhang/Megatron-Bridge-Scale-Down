#!/usr/bin/env node
// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "../../../../..");
const DEFAULT_RESULTS_ROOT = path.join(
  REPO_ROOT,
  "results/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model",
);
const SAMPLE_STEPS = [4, 5, 6, 7, 8]; // Zero-based keys for iterations 5-9.
const RUN_NAME_PATTERN =
  /^(dense|expert)-(default|alltoall|hybridep)-(baseline|offload)(?:-mbs(\d+))?-r\d+$/;
const HEADERS = [
  "model",
  "dispatcher",
  "mbs",
  "dtype",
  "type",
  "baseline (TFlops)",
  "offload (TFlops)",
  "offload 相对于 baseline 的性能下降幅度",
];

function usage() {
  process.stdout.write(`Usage: analyse_mlp_offload_results.mjs --model <qwen|deepseek> [OPTIONS]

Read successful non-profiled runs, select the latest baseline and offload for
each model/dispatcher/MBS/dtype/type combination, and write an XLSX throughput
comparison using iterations 5-9.

Options:
  --model <qwen|deepseek>  Model family to analyse (required)
  --results-root <path>    Raw result tree (default: ${DEFAULT_RESULTS_ROOT})
  --output <path>          XLSX path (default: <results-root>/offload-throughput-<model>.xlsx)
  --artifact-tool <path>   Path or module specifier for @oai/artifact-tool
  --dry-run                Parse and validate data, then print JSON without writing XLSX
  -h, --help               Show this help

The XLSX writer requires @oai/artifact-tool. Set ARTIFACT_TOOL_MODULE or pass
--artifact-tool when it is not available through Node's normal module lookup.
`);
}

function parseArgs(argv) {
  const options = {
    modelFamily: null,
    resultsRoot: DEFAULT_RESULTS_ROOT,
    output: null,
    artifactTool: process.env.ARTIFACT_TOOL_MODULE ?? "@oai/artifact-tool",
    dryRun: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "-h" || argument === "--help") {
      usage();
      process.exit(0);
    }
    if (argument === "--dry-run") {
      options.dryRun = true;
      continue;
    }
    const value = argv[index + 1];
    if (value === undefined) {
      throw new Error(`Missing value for ${argument}`);
    }
    if (argument === "--model") {
      options.modelFamily = value;
    } else if (argument === "--results-root") {
      options.resultsRoot = path.resolve(value);
    } else if (argument === "--output") {
      options.output = path.resolve(value);
    } else if (argument === "--artifact-tool") {
      options.artifactTool = value;
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
    index += 1;
  }

  if (options.modelFamily !== "qwen" && options.modelFamily !== "deepseek") {
    throw new Error("--model must be qwen or deepseek");
  }
  return options;
}

async function walkConfigFiles(directory) {
  const configPaths = [];
  let entries;
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") {
      return configPaths;
    }
    throw error;
  }

  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      configPaths.push(...(await walkConfigFiles(entryPath)));
    } else if (entry.isFile() && entry.name === "config.json") {
      configPaths.push(entryPath);
    }
  }
  return configPaths;
}

async function readJson(jsonPath) {
  return JSON.parse(await fs.readFile(jsonPath, "utf8"));
}

function matchesModelFamily(model, modelFamily) {
  const normalized = model.toLowerCase();
  return modelFamily === "qwen" ? normalized.startsWith("qwen") : normalized.startsWith("deepseek");
}

function parseIdentity(config, configPath) {
  const runName = String(config.run_name ?? "");
  const match = RUN_NAME_PATTERN.exec(runName);
  if (match === null) {
    throw new Error(`Unrecognized run_name in ${configPath}: ${runName}`);
  }

  const microBatchSize = Number(config.micro_batch_size ?? match[4]);
  const runNameMbs = match[4] === undefined ? null : Number(match[4]);
  if (!Number.isInteger(microBatchSize) || microBatchSize <= 0) {
    throw new Error(`Invalid micro_batch_size in ${configPath}`);
  }
  if (runNameMbs !== null && runNameMbs !== microBatchSize) {
    throw new Error(`run_name and config disagree on micro_batch_size in ${configPath}`);
  }

  const dispatcher = String(config.dispatcher ?? match[2]);
  if (dispatcher !== match[2]) {
    throw new Error(`run_name and config disagree on dispatcher in ${configPath}`);
  }
  const recordedDtype = String(config.dtype ?? config.precision ?? "");
  const dtype = recordedDtype === "fp8mx" ? "mxfp8" : recordedDtype;
  if (dtype !== "bf16" && dtype !== "mxfp8") {
    throw new Error(`Invalid dtype in ${configPath}: ${recordedDtype}`);
  }

  const model = String(config.model ?? "");
  const runTime = String(config.run_time ?? "");
  if (model.length === 0 || runTime.length === 0) {
    throw new Error(`Missing model or run_time in ${configPath}`);
  }
  return {
    model,
    dispatcher,
    mbs: microBatchSize,
    dtype,
    type: match[1],
    caseName: match[3],
    runTime,
  };
}

async function isSuccessfulRun(resultDir) {
  try {
    const summary = await readJson(path.join(resultDir, "summary.json"));
    return Number(summary.status) === 0;
  } catch (error) {
    if (error.code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

async function readMeanTflops(resultDir) {
  const metricsPath = path.join(resultDir, "gpu_utilization.json");
  const metrics = await readJson(metricsPath);
  const values = SAMPLE_STEPS.map((step) => Number(metrics[String(step)]));
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${metricsPath} is missing finite TFlops values for iterations 5-9`);
  }
  return { mean: mean(values), samples: values };
}

function groupKey(identity) {
  return [identity.model, identity.dispatcher, identity.mbs, identity.dtype, identity.type].join("/");
}

function sortRows(left, right) {
  const dispatcherOrder = { default: 0, alltoall: 1, hybridep: 2 };
  const typeOrder = { dense: 0, expert: 1 };
  return (
    dispatcherOrder[left.dispatcher] - dispatcherOrder[right.dispatcher] ||
    left.mbs - right.mbs ||
    left.dtype.localeCompare(right.dtype) ||
    typeOrder[left.type] - typeOrder[right.type] ||
    left.model.localeCompare(right.model)
  );
}

async function discoverRows(resultsRoot, modelFamily) {
  const configPaths = await walkConfigFiles(resultsRoot);
  const groups = new Map();

  for (const configPath of configPaths) {
    const config = await readJson(configPath);
    if ((config.profile ?? "none") !== "none") {
      continue;
    }
    const model = String(config.model ?? "");
    if (!matchesModelFamily(model, modelFamily)) {
      continue;
    }

    const resultDir = path.dirname(configPath);
    if (!(await isSuccessfulRun(resultDir))) {
      continue;
    }
    const identity = parseIdentity(config, configPath);
    const key = groupKey(identity);
    if (!groups.has(key)) {
      groups.set(key, { baseline: new Map(), offload: new Map() });
    }
    const caseRuns = groups.get(key)[identity.caseName];
    if (caseRuns.has(identity.runTime)) {
      throw new Error(`Duplicate ${identity.caseName} run for ${key} at ${identity.runTime}`);
    }
    caseRuns.set(identity.runTime, { identity, resultDir });
  }

  const rows = [];
  for (const [key, caseRuns] of groups) {
    if (caseRuns.baseline.size === 0 || caseRuns.offload.size === 0) {
      throw new Error(`Missing baseline or offload run for ${key}`);
    }

    const baselineRunTime = [...caseRuns.baseline.keys()].sort().at(-1);
    const offloadRunTime = [...caseRuns.offload.keys()].sort().at(-1);
    const baselineRun = caseRuns.baseline.get(baselineRunTime);
    const offloadRun = caseRuns.offload.get(offloadRunTime);
    const baseline = await readMeanTflops(baselineRun.resultDir);
    const offload = await readMeanTflops(offloadRun.resultDir);
    if (baseline.mean <= 0) {
      throw new Error(`Baseline mean TFlops must be positive for ${key} at ${baselineRunTime}`);
    }
    rows.push({
      model: baselineRun.identity.model,
      dispatcher: baselineRun.identity.dispatcher,
      mbs: baselineRun.identity.mbs,
      dtype: baselineRun.identity.dtype,
      type: baselineRun.identity.type,
      baselineTflops: baseline.mean,
      offloadTflops: offload.mean,
      performanceDrop: (baseline.mean - offload.mean) / baseline.mean,
      baselineRunTime,
      offloadRunTime,
      baselineSamples: baseline.samples,
      offloadSamples: offload.samples,
    });
  }

  if (rows.length === 0) {
    throw new Error(`No successful non-profiled ${modelFamily} runs found under ${resultsRoot}`);
  }
  return rows.sort(sortRows);
}

function asImportSpecifier(specifier) {
  if (specifier.startsWith(".") || specifier.startsWith("/")) {
    return pathToFileURL(path.resolve(specifier)).href;
  }
  return specifier;
}

async function loadArtifactTool(specifier) {
  try {
    return await import(asImportSpecifier(specifier));
  } catch (error) {
    throw new Error(
      `Unable to load @oai/artifact-tool from ${specifier}. ` +
      "Install it for Node or pass --artifact-tool/ARTIFACT_TOOL_MODULE with its module path.",
      { cause: error },
    );
  }
}

async function writeWorkbook(rows, outputPath, artifactToolSpecifier) {
  const { SpreadsheetFile, Workbook } = await loadArtifactTool(artifactToolSpecifier);
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Throughput");
  const lastRow = rows.length + 1;
  const tableRange = `A1:H${lastRow}`;

  sheet.showGridLines = false;
  sheet.tabColor = "#1F4E78";
  sheet.getRange("A1:H1").values = [HEADERS];
  sheet.getRange(`A2:H${lastRow}`).values = rows.map((row) => [
    row.model,
    row.dispatcher,
    row.mbs,
    row.dtype,
    row.type,
    row.baselineTflops,
    row.offloadTflops,
    row.performanceDrop,
  ]);

  sheet.getRange(tableRange).format.font = { name: "Arial", size: 10, color: "#1F1F1F" };
  sheet.getRange("A1:H1").format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 30,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  sheet.getRange(`A2:H${lastRow}`).format.verticalAlignment = "center";
  sheet.getRange(`B2:E${lastRow}`).format.horizontalAlignment = "center";
  sheet.getRange(`C2:C${lastRow}`).format.numberFormat = "0";
  sheet.getRange(`F2:G${lastRow}`).format.numberFormat = "0.00";
  sheet.getRange(`H2:H${lastRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`A2:H${lastRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E1F2" },
    insideVertical: { style: "thin", color: "#E5E7EB" },
    bottom: { style: "thin", color: "#A6A6A6" },
  };
  sheet.getRange(`H2:H${lastRow}`).conditionalFormats.add("cellIs", {
    operator: "greaterThan",
    formula: 0,
    format: { fill: "#FCE8E6", font: { color: "#B91C1C" } },
  });
  sheet.getRange(`H2:H${lastRow}`).conditionalFormats.add("cellIs", {
    operator: "lessThan",
    formula: 0,
    format: { fill: "#E6F4EA", font: { color: "#166534" } },
  });
  sheet.freezePanes.freezeRows(1);

  for (const [column, width] of [
    ["A", 26], ["B", 14], ["C", 8], ["D", 10], ["E", 10], ["F", 20], ["G", 20], ["H", 38],
  ]) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }

  workbook.recalculate();
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);
  await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const rows = await discoverRows(options.resultsRoot, options.modelFamily);
  const outputPath = options.output ?? path.join(
    options.resultsRoot,
    `offload-throughput-${options.modelFamily}.xlsx`,
  );

  if (options.dryRun) {
    process.stdout.write(`${JSON.stringify({
      modelFamily: options.modelFamily,
      outputPath,
      headers: HEADERS,
      rows,
    }, null, 2)}\n`);
    return;
  }

  await writeWorkbook(rows, outputPath, options.artifactTool);
  process.stdout.write(`Wrote ${rows.length} throughput comparison row(s): ${outputPath}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
