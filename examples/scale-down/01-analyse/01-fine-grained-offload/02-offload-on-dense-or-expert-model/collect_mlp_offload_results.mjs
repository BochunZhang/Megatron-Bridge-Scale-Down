#!/usr/bin/env node
// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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
  "result/01-analyse/01-fine-grained-offload/02-offload-on-dense-or-expert-model",
);
const SAMPLE_ITERATIONS = [5, 6, 7, 8, 9];
const RUN_NAME_PATTERN =
  /^(dense|expert)-(default|alltoall|hybridep)-(baseline|offload)(?:-mbs(\d+))?-r(\d+)$/;
const ITERATION_PATTERN =
  /iteration\s+(\d+)\s*\/\s*\d+\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+)/i;

function usage() {
  process.stdout.write(`Usage: collect_mlp_offload_results.mjs [OPTIONS]

Collect iterations 5-9 from successful, non-profiled MLP offload runs and
write an XLSX comparison workbook.

Options:
  --results-root <path>  Raw result tree (default: ${DEFAULT_RESULTS_ROOT})
  --run-time <id>        Batch id to collect (default: latest non-profiled batch)
  --output <path>        XLSX path (default: <results-root>/offload-comparison-<id>.xlsx)
  --artifact-tool <path> Path or module specifier for @oai/artifact-tool
  --dry-run              Parse and validate data, then print JSON without writing XLSX
  -h, --help             Show this help

The XLSX writer requires @oai/artifact-tool. Set ARTIFACT_TOOL_MODULE or pass
--artifact-tool when it is not available through Node's normal module lookup.
`);
}

function parseArgs(argv) {
  const options = {
    resultsRoot: DEFAULT_RESULTS_ROOT,
    runTime: null,
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
    if (argument === "--results-root") {
      options.resultsRoot = path.resolve(value);
    } else if (argument === "--run-time") {
      options.runTime = value;
    } else if (argument === "--output") {
      options.output = path.resolve(value);
    } else if (argument === "--artifact-tool") {
      options.artifactTool = value;
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
    index += 1;
  }
  return options;
}

async function walk(directory) {
  const paths = [];
  let entries;
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") {
      return paths;
    }
    throw error;
  }
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      paths.push(...(await walk(entryPath)));
    } else if (entry.isFile() && entry.name === "config.json") {
      paths.push(entryPath);
    }
  }
  return paths;
}

async function readJson(jsonPath) {
  return JSON.parse(await fs.readFile(jsonPath, "utf8"));
}

function parseIdentity(runName) {
  const match = RUN_NAME_PATTERN.exec(runName);
  if (match === null) {
    throw new Error(`Unrecognized run name: ${runName}`);
  }
  return {
    modelKind: match[1],
    dispatcherFromName: match[2],
    caseName: match[3],
    microBatchSizeFromName: match[4] === undefined ? null : Number(match[4]),
    repeat: Number(match[5]),
  };
}

function parseIterationTimes(logText) {
  const times = new Map();
  for (const line of logText.replaceAll(/\x1b\[[0-9;]*m/g, "").split(/\r?\n/)) {
    const match = ITERATION_PATTERN.exec(line);
    if (match === null) {
      continue;
    }
    const iteration = Number(match[1]);
    const stepTimeMs = Number(match[2]);
    if (times.has(iteration) && times.get(iteration) !== stepTimeMs) {
      throw new Error(`Iteration ${iteration} has conflicting step times in train.log`);
    }
    times.set(iteration, stepTimeMs);
  }
  return times;
}

async function parseRun(configPath) {
  const config = await readJson(configPath);
  if ((config.profile ?? "none") !== "none") {
    return null;
  }

  const resultDir = path.dirname(configPath);
  const identity = parseIdentity(String(config.run_name));
  const summaryPath = path.join(resultDir, "summary.json");
  const summary = await readJson(summaryPath).catch(() => ({ status: 1 }));
  if (Number(summary.status) !== 0) {
    throw new Error(`Training did not complete successfully: ${resultDir}`);
  }

  const logText = await fs.readFile(path.join(resultDir, "train.log"), "utf8");
  const iterationTimes = parseIterationTimes(logText);
  const missing = SAMPLE_ITERATIONS.filter((iteration) => !iterationTimes.has(iteration));
  if (missing.length > 0) {
    throw new Error(`${resultDir} is missing iteration(s): ${missing.join(", ")}`);
  }

  const globalBatchSize = Number(config.global_batch_size);
  const microBatchSize = Number(config.micro_batch_size);
  const sequenceLength = Number(config.sequence_length);
  const recordedDtype = String(config.dtype ?? config.precision);
  const dtype = recordedDtype === "fp8mx" ? "mxfp8" : recordedDtype;
  if (!(globalBatchSize > 0) || !(microBatchSize > 0) || !(sequenceLength > 0)) {
    throw new Error(`Invalid batch size or sequence_length in ${configPath}`);
  }
  if (dtype !== "bf16" && dtype !== "mxfp8") {
    throw new Error(`Invalid or missing dtype in ${configPath}`);
  }
  if (
    identity.microBatchSizeFromName !== null &&
    identity.microBatchSizeFromName !== microBatchSize
  ) {
    throw new Error(`Run name and config disagree on micro batch size: ${configPath}`);
  }

  const samples = SAMPLE_ITERATIONS.map((iteration) => {
    const stepTimeMs = iterationTimes.get(iteration);
    return {
      iteration,
      stepTimeMs,
      tokensPerSecond: (globalBatchSize * sequenceLength * 1000) / stepTimeMs,
    };
  });
  return {
    runTime: String(config.run_time),
    modelKind: identity.modelKind,
    model: String(config.model),
    dtype,
    dispatcher: String(config.dispatcher ?? identity.dispatcherFromName),
    caseName: identity.caseName,
    repeat: identity.repeat,
    globalBatchSize,
    microBatchSize,
    sequenceLength,
    resultDir,
    samples,
  };
}

function sortRuns(left, right) {
  const modelOrder = { dense: 0, expert: 1 };
  const dispatcherOrder = { default: 0, alltoall: 1, hybridep: 2 };
  const caseOrder = { baseline: 0, offload: 1 };
  return (
    modelOrder[left.modelKind] - modelOrder[right.modelKind] ||
    dispatcherOrder[left.dispatcher] - dispatcherOrder[right.dispatcher] ||
    left.microBatchSize - right.microBatchSize ||
    caseOrder[left.caseName] - caseOrder[right.caseName] ||
    left.repeat - right.repeat
  );
}

async function discoverRuns(resultsRoot, requestedRunTime) {
  const configPaths = await walk(resultsRoot);
  const candidates = [];
  for (const configPath of configPaths) {
    const config = await readJson(configPath);
    if ((config.profile ?? "none") !== "none") {
      continue;
    }
    if (!RUN_NAME_PATTERN.test(String(config.run_name))) {
      continue;
    }
    candidates.push({ configPath, runTime: String(config.run_time) });
  }
  if (candidates.length === 0) {
    throw new Error(`No successful non-profiled runs found under ${resultsRoot}`);
  }

  const runTimes = [...new Set(candidates.map((candidate) => candidate.runTime))].sort();
  const runTime = requestedRunTime ?? runTimes.at(-1);
  const selectedPaths = candidates
    .filter((candidate) => candidate.runTime === runTime)
    .map((candidate) => candidate.configPath);
  const selected = (await Promise.all(selectedPaths.map((configPath) => parseRun(configPath))))
    .filter(Boolean)
    .sort(sortRuns);
  if (selected.length === 0) {
    throw new Error(`No successful non-profiled runs found for run_time=${runTime}`);
  }

  const identities = new Set();
  for (const run of selected) {
    const identity = `${run.modelKind}/${run.model}/${run.dtype}/${run.dispatcher}/${run.caseName}/mbs${run.microBatchSize}/r${run.repeat}`;
    if (identities.has(identity)) {
      throw new Error(`Duplicate run identity for run_time=${runTime}: ${identity}`);
    }
    identities.add(identity);
  }
  return { runTime, runs: selected };
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

function setColumnWidths(sheet, widths) {
  widths.forEach(([column, width]) => {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  });
}

async function writeWorkbook(runs, runTime, outputPath, artifactToolSpecifier) {
  const { SpreadsheetFile, Workbook } = await loadArtifactTool(artifactToolSpecifier);
  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const samples = workbook.worksheets.add("Samples");

  summary.showGridLines = false;
  samples.showGridLines = false;
  summary.tabColor = "#1F4E78";
  samples.tabColor = "#70AD47";

  summary.getRange("A1:T1").merge();
  summary.getRange("A1").values = [["MLP Offload Throughput Comparison"]];
  summary.getRange("A2:T2").merge();
  summary.getRange("A2").values = [[
    `run_time=${runTime}; successful profile=none runs only; throughput uses iterations 5-9`,
  ]];
  summary.getRange("A1:T1").format.fill = "#1F4E78";
  summary.getRange("A1:T1").format.font = { bold: true, color: "#FFFFFF", size: 16 };
  summary.getRange("A1:T1").format.rowHeight = 28;
  summary.getRange("A2:T2").format.fill = "#D9EAF7";
  summary.getRange("A2:T2").format.font = { color: "#1F1F1F", italic: true };

  const summaryHeaders = [
    "Model Type",
    "Model",
    "DType",
    "Dispatcher",
    "Case",
    "Repeat",
    "MBS",
    "Iter 5",
    "Iter 6",
    "Iter 7",
    "Iter 8",
    "Iter 9",
    "Mean",
    "Median",
    "Min",
    "Max",
    "Std Dev",
    "CV",
    "vs Baseline",
    "vs All-to-All",
  ];
  summary.getRange("A4:T4").values = [summaryHeaders];
  summary.getRange("A4:T4").format.fill = "#4472C4";
  summary.getRange("A4:T4").format.font = { bold: true, color: "#FFFFFF" };
  summary.getRange("A4:T4").format.wrapText = true;
  summary.getRange("A4:T4").format.rowHeight = 30;

  const firstDataRow = 5;
  const baselineRows = new Map();
  const alltoallRows = new Map();
  runs.forEach((run, index) => {
    const row = firstDataRow + index;
    const baselineKey =
      `${run.modelKind}/${run.model}/${run.dtype}/${run.dispatcher}/mbs${run.microBatchSize}/${run.repeat}`;
    const alltoallKey =
      `${run.modelKind}/${run.model}/${run.dtype}/${run.caseName}/mbs${run.microBatchSize}/${run.repeat}`;
    if (run.caseName === "baseline") {
      baselineRows.set(baselineKey, row);
    }
    if (run.dispatcher === "alltoall") {
      alltoallRows.set(alltoallKey, row);
    }
  });

  const summaryValues = runs.map((run) => [
    run.modelKind,
    run.model,
    run.dtype,
    run.dispatcher,
    run.caseName,
    run.repeat,
    run.microBatchSize,
    ...run.samples.map((sample) => sample.tokensPerSecond),
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
  ]);
  if (summaryValues.length > 0) {
    summary.getRangeByIndexes(firstDataRow - 1, 0, summaryValues.length, summaryHeaders.length).values = summaryValues;
  }

  runs.forEach((run, index) => {
    const row = firstDataRow + index;
    summary.getRange(`M${row}`).formulas = [[`=AVERAGE(H${row}:L${row})`]];
    summary.getRange(`N${row}`).formulas = [[`=MEDIAN(H${row}:L${row})`]];
    summary.getRange(`O${row}`).formulas = [[`=MIN(H${row}:L${row})`]];
    summary.getRange(`P${row}`).formulas = [[`=MAX(H${row}:L${row})`]];
    summary.getRange(`Q${row}`).formulas = [[`=STDEV.S(H${row}:L${row})`]];
    summary.getRange(`R${row}`).formulas = [[`=IFERROR(Q${row}/M${row},"")`]];

    const baselineKey =
      `${run.modelKind}/${run.model}/${run.dtype}/${run.dispatcher}/mbs${run.microBatchSize}/${run.repeat}`;
    const baselineRow = baselineRows.get(baselineKey);
    summary.getRange(`S${row}`).formulas = [[
      baselineRow === undefined ? '=""' : `=IFERROR(M${row}/M${baselineRow}-1,"")`,
    ]];

    const alltoallKey =
      `${run.modelKind}/${run.model}/${run.dtype}/${run.caseName}/mbs${run.microBatchSize}/${run.repeat}`;
    const alltoallRow = alltoallRows.get(alltoallKey);
    summary.getRange(`T${row}`).formulas = [[
      run.modelKind !== "expert" || alltoallRow === undefined
        ? '=""'
        : `=IFERROR(M${row}/M${alltoallRow}-1,"")`,
    ]];
  });

  const summaryLastRow = firstDataRow + runs.length - 1;
  summary.getRange(`H${firstDataRow}:Q${summaryLastRow}`).format.numberFormat = "#,##0.00";
  summary.getRange(`R${firstDataRow}:T${summaryLastRow}`).format.numberFormat = "0.00%";
  summary.getRange(`A${firstDataRow}:T${summaryLastRow}`).format.borders = {
    top: { style: "continuous", color: "#D9E1F2" },
    bottom: { style: "continuous", color: "#D9E1F2" },
  };
  summary.getRange(`A${firstDataRow}:T${summaryLastRow}`).format.verticalAlignment = "center";
  summary.freezePanes.freezeRows(4);
  setColumnWidths(summary, [
    ["A", 12], ["B", 24], ["C", 10], ["D", 13], ["E", 11], ["F", 8], ["G", 8],
    ["H", 13], ["I", 13], ["J", 13], ["K", 13], ["L", 13], ["M", 14],
    ["N", 14], ["O", 14], ["P", 14], ["Q", 14], ["R", 11], ["S", 14], ["T", 15],
  ]);

  const sampleHeaders = [
    "Run Time",
    "Model Type",
    "Model",
    "DType",
    "Dispatcher",
    "Case",
    "Repeat",
    "MBS",
    "Iteration",
    "Step Time (ms)",
    "Global Batch Size",
    "Sequence Length",
    "Throughput (tokens/s)",
    "Result Directory",
  ];
  const sampleRows = runs.flatMap((run) =>
    run.samples.map((sample) => [
      run.runTime,
      run.modelKind,
      run.model,
      run.dtype,
      run.dispatcher,
      run.caseName,
      run.repeat,
      run.microBatchSize,
      sample.iteration,
      sample.stepTimeMs,
      run.globalBatchSize,
      run.sequenceLength,
      sample.tokensPerSecond,
      run.resultDir,
    ]),
  );
  samples.getRange("A1:N1").values = [sampleHeaders];
  samples.getRange("A1:N1").format.fill = "#548235";
  samples.getRange("A1:N1").format.font = { bold: true, color: "#FFFFFF" };
  samples.getRange("A1:N1").format.wrapText = true;
  samples.getRange("A1:N1").format.rowHeight = 30;
  samples.getRangeByIndexes(1, 0, sampleRows.length, sampleHeaders.length).values = sampleRows;
  samples.getRange(`J2:J${sampleRows.length + 1}`).format.numberFormat = "#,##0.0";
  samples.getRange(`M2:M${sampleRows.length + 1}`).format.numberFormat = "#,##0.00";
  samples.freezePanes.freezeRows(1);
  setColumnWidths(samples, [
    ["A", 20], ["B", 12], ["C", 24], ["D", 10], ["E", 13], ["F", 11],
    ["G", 8], ["H", 8], ["I", 10], ["J", 16], ["K", 17], ["L", 16], ["M", 22], ["N", 72],
  ]);

  await workbook.recalculate();
  const formulaInspection = await workbook.inspect({
    kind: "formula",
    sheetId: "Summary",
    range: `M${firstDataRow}:T${summaryLastRow}`,
    maxChars: 5000,
  });
  const errorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    maxChars: 5000,
  });
  const errorInspectionText = errorInspection.ndjson ?? JSON.stringify(errorInspection);
  if (errorInspectionText.match(/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/)) {
    throw new Error(`Formula verification failed: ${errorInspectionText}`);
  }
  const formulaInspectionText = formulaInspection.ndjson ?? JSON.stringify(formulaInspection);
  const writtenFormulas = summary.getRange(`M${firstDataRow}:T${summaryLastRow}`).formulas;
  if (!formulaInspectionText.includes("AVERAGE") && !JSON.stringify(writtenFormulas).includes("AVERAGE")) {
    throw new Error("Formula verification did not find the expected summary formulas");
  }

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const { runTime, runs } = await discoverRuns(options.resultsRoot, options.runTime);
  const outputPath = options.output ?? path.join(options.resultsRoot, `offload-comparison-${runTime}.xlsx`);

  if (options.dryRun) {
    process.stdout.write(`${JSON.stringify({ runTime, outputPath, runs }, null, 2)}\n`);
    return;
  }

  await writeWorkbook(runs, runTime, outputPath, options.artifactTool);
  process.stdout.write(`Collected ${runs.length} group(s), 5 iterations per group: ${outputPath}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
