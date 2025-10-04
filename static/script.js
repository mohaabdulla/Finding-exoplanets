const sampleData = {
  generated_at: "2025-10-04T12:00:00Z",
  summary: {
    average_precision: 0.8888,
    total_samples: 1913,
    filtered_samples: 1772,
    confirmed_predictions: 469,
    hz_candidates: 3,
    anomalies_detected: 141,
    f1_score: 0.812,
    precision: 0.775,
    recall: 0.853,
    accuracy: 0.887,
    positive_rate: 0.291,
  },
  model_overview: {
    name: "HistGradientBoostingClassifier",
    metrics: {
      average_precision: 0.8888,
      f1_score: 0.812,
      precision: 0.775,
      recall: 0.853,
      accuracy: 0.887,
      threshold: 0.327,
      threshold_hz: 0.327,
    },
  },
  performance: {
    models: ["HistGradientBoostingClassifier"],
    ap_scores: [0.8888],
    f1_scores: [0.812],
    accuracy_scores: [0.887],
    precision_scores: [0.775],
    recall_scores: [0.853],
    thresholds: [0.327],
    thresholds_hz: [0.327],
  },
  hz_planets: {
    true: ["Kepler-452 b", "Kepler-712 c", "Kepler-440 b"],
    predicted: ["Kepler-440 b"],
    counts: {
      true_planets: 610,
      predicted_planets: 469,
      true_hz_planets: 3,
      predicted_hz_planets: 1,
    },
  },
  confusion_matrix: {
    true_negative: 1072,
    false_positive: 141,
    false_negative: 77,
    true_positive: 469,
  },
  training_info: {
    best_model: "HistGradientBoostingClassifier",
    threshold: 0.327,
    threshold_hz: 0.327,
    train_samples: 7651,
    test_samples: 1913,
    filtered_samples: 1772,
    target_distribution: {
      class_0: 6818,
      class_1: 2746,
    },
  },
  predictions: [
    {
      name: "Kepler-452 b",
      prediction: "CONFIRMED",
      actual: "CONFIRMED",
      confidence: 0.9,
      hz: true,
      anomaly: false,
    },
    {
      name: "Kepler-712 c",
      prediction: "CONFIRMED",
      actual: "CONFIRMED",
      confidence: 0.85,
      hz: true,
      anomaly: false,
    },
    {
      name: "Kepler-440 b",
      prediction: "CONFIRMED",
      actual: "CONFIRMED",
      confidence: 0.88,
      hz: true,
      anomaly: false,
    },
    {
      name: "Kepler-186f",
      prediction: "NOT_CONFIRMED",
      actual: "CONFIRMED",
      confidence: 0.41,
      hz: true,
      anomaly: false,
    },
    {
      name: "KIC 10255705",
      prediction: "NOT_CONFIRMED",
      actual: "NOT_CONFIRMED",
      confidence: 0.15,
      hz: false,
      anomaly: false,
    },
  ],
};

let performanceChart;

function asNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function formatDecimal(value, digits = 3) {
  return asNumber(value).toFixed(digits);
}

function formatPercent(value, digits = 1) {
  return `${(asNumber(value) * 100).toFixed(digits)}%`;
}

async function loadData() {
  try {
    const response = await fetch("/results.json", { cache: "no-store" });
    if (response.ok) {
      return await response.json();
    }
    console.warn("results.json not available, using sample data");
    return sampleData;
  } catch (error) {
    console.warn("Error loading results, using sample data", error);
    return sampleData;
  }
}

function updateSummary(summary, metrics) {
  const averagePrecision = summary.average_precision ?? metrics.average_precision ?? 0;
  document.getElementById("apValue").textContent = formatDecimal(averagePrecision);
  document.getElementById("totalSamples").textContent = asNumber(summary.total_samples).toLocaleString();
  document.getElementById("confirmedValue").textContent = asNumber(summary.confirmed_predictions).toLocaleString();
  document.getElementById("hzValue").textContent = asNumber(summary.hz_candidates).toLocaleString();
  document.getElementById("anomaliesValue").textContent = asNumber(summary.anomalies_detected).toLocaleString();

  const f1 = summary.f1_score ?? metrics.f1_score ?? 0;
  document.getElementById("f1Value").textContent = formatDecimal(f1);
}

function updateModelOverview(model) {
  const metrics = model.metrics || {};
  document.getElementById("bestModelSummary").textContent = `Model: ${model.name || "--"}`;
  document.getElementById("bestModelName").textContent = model.name || "--";
  document.getElementById("modelPrecision").textContent = formatDecimal(metrics.precision);
  document.getElementById("modelRecall").textContent = formatDecimal(metrics.recall);
  document.getElementById("modelAccuracy").textContent = formatDecimal(metrics.accuracy);
  document.getElementById("modelAP").textContent = formatDecimal(metrics.average_precision);
  document.getElementById("modelThreshold").textContent = formatDecimal(metrics.threshold, 3);
  document.getElementById("modelThresholdHz").textContent = formatDecimal(metrics.threshold_hz, 3);
}

function updateHabitableZones(hzPlanets) {
  const trueHzContainer = document.getElementById("trueHzPlanets");
  const predictedHzContainer = document.getElementById("predictedHzPlanets");
  trueHzContainer.innerHTML = "";
  predictedHzContainer.innerHTML = "";

  (hzPlanets?.true || []).forEach((planet) => {
    const planetItem = document.createElement("div");
    planetItem.className = "planet-item";
    planetItem.innerHTML = `<i class="fas fa-globe-americas"></i> ${planet}`;
    trueHzContainer.appendChild(planetItem);
  });

  (hzPlanets?.predicted || []).forEach((planet) => {
    const planetItem = document.createElement("div");
    planetItem.className = "planet-item";
    planetItem.innerHTML = `<i class="fas fa-search"></i> ${planet}`;
    predictedHzContainer.appendChild(planetItem);
  });
}

function updatePredictionsTable(predictions) {
  const tableBody = document.getElementById("resultsTableBody");
  tableBody.innerHTML = "";

  if (!Array.isArray(predictions)) {
    return;
  }

  predictions.forEach((pred) => {
    const row = document.createElement("tr");
    const statusClass = pred.prediction === "CONFIRMED" ? "status-confirmed" : "status-not-confirmed";
    const anomalyClass = pred.anomaly ? "status-badge status-anomaly" : "";
    const hzIcon = pred.hz ? '<i class="fas fa-sun" style="color: #f39c12;"></i>' : "";
    const confidence = pred.confidence != null ? `${formatPercent(pred.confidence, 1)}` : "--";

    row.innerHTML = `
      <td>${pred.name || "--"}</td>
      <td><span class="status-badge ${statusClass}">${pred.prediction || "--"}</span></td>
      <td>${pred.actual || "--"}</td>
      <td>${confidence}</td>
      <td>${hzIcon}</td>
      <td>${pred.anomaly ? '<span class="status-badge status-anomaly">ANOMALY</span>' : ""}</td>
    `;

    tableBody.appendChild(row);
  });
}

function updateLastUpdated(timestamp) {
  const node = document.getElementById("lastUpdated");
  if (!node) return;
  if (!timestamp) {
    node.textContent = `Last updated: ${new Date().toLocaleString()}`;
    return;
  }
  const dt = new Date(timestamp);
  if (Number.isNaN(dt.getTime())) {
    node.textContent = `Last updated: ${new Date().toLocaleString()}`;
  } else {
    node.textContent = `Last updated: ${dt.toLocaleString()}`;
  }
}

function createPerformanceChart(performance) {
  const ctx = document.getElementById("performanceChart").getContext("2d");
  if (performanceChart) {
    performanceChart.destroy();
    performanceChart = null;
  }

  const labels = performance?.models || [];
  if (labels.length === 0) {
    return;
  }

  const datasets = [
    {
      label: "Average Precision",
      data: performance.ap_scores || [],
      backgroundColor: "#3498db",
      borderColor: "#2980b9",
      borderWidth: 1,
    },
    {
      label: "F1 Score",
      data: performance.f1_scores || [],
      backgroundColor: "#9b59b6",
      borderColor: "#8e44ad",
      borderWidth: 1,
    },
  ];

  if (performance.accuracy_scores && performance.accuracy_scores.length === labels.length) {
    datasets.push({
      label: "Accuracy",
      data: performance.accuracy_scores,
      backgroundColor: "#2ecc71",
      borderColor: "#27ae60",
      borderWidth: 1,
    });
  }

  performanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets,
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          title: {
            display: true,
            text: "Score",
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: "Model Performance Comparison",
        },
      },
    },
  });
}

async function initializeDashboard() {
  const data = await loadData();
  const summary = data.summary || {};
  const modelOverview = data.model_overview || { metrics: {} };

  updateSummary(summary, modelOverview.metrics || {});
  updateModelOverview(modelOverview);
  updateHabitableZones(data.hz_planets || {});
  updatePredictionsTable(data.predictions);
  updateLastUpdated(data.generated_at);
  createPerformanceChart(data.performance);
}

window.addEventListener("DOMContentLoaded", initializeDashboard);
