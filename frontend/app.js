/**
 * frontend/app.js
 * ----------------
 * Frontend controller connecting clinical UI to Flask backend API:
 * - GET  /api/health   : Health status & model metrics
 * - GET  /api/samples  : Preset test cases for 1-click viva demos
 * - POST /api/predict  : Quick Random Forest classification
 * - POST /api/explain  : Real DiCE counterfactual explanations with delta diffs
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Status & Metrics
  const systemStatus = document.getElementById('systemStatus');
  const statusText = document.getElementById('statusText');
  const metaModel = document.getElementById('metaModel');
  const metaAccuracy = document.getElementById('metaAccuracy');
  const metaRoc = document.getElementById('metaRoc');
  const metaPR = document.getElementById('metaPR');
  const metaXai = document.getElementById('metaXai');

  // DOM Elements - Presets & Form
  const btnLoadPositive = document.getElementById('btnLoadPositive');
  const btnLoadNegative = document.getElementById('btnLoadNegative');
  const btnResetForm = document.getElementById('btnResetForm');
  const patientForm = document.getElementById('patientForm');
  const btnQuickPredict = document.getElementById('btnQuickPredict');

  // DOM Elements - Results Display
  const emptyState = document.getElementById('emptyState');
  const loadingState = document.getElementById('loadingState');
  const loadingTitle = document.getElementById('loadingTitle');
  const loadingMsg = document.getElementById('loadingMsg');
  const resultContent = document.getElementById('resultContent');

  const outcomeCard = document.getElementById('outcomeCard');
  const outcomeBadge = document.getElementById('outcomeBadge');
  const confidenceValue = document.getElementById('confidenceValue');
  const confidenceBar = document.getElementById('confidenceBar');
  const probHealthy = document.getElementById('probHealthy');
  const probDisease = document.getElementById('probDisease');

  const xaiContainer = document.getElementById('xaiContainer');
  const xaiSubtitle = document.getElementById('xaiSubtitle');
  const cfListContainer = document.getElementById('cfListContainer');

  // DOM Elements - Tabs
  const tabClinical = document.getElementById('tabClinical');
  const navXai = document.getElementById('navXai');
  const tabAdmin = document.getElementById('tabAdmin');
  const clinicalView = document.getElementById('clinicalView');
  const adminView = document.getElementById('adminView');

  // DOM Elements - Admin Module 1 (Dataset)
  const datasetFileInput = document.getElementById('datasetFileInput');
  const uploadLabelText = document.getElementById('uploadLabelText');
  const btnUploadDataset = document.getElementById('btnUploadDataset');
  const btnInspectDefault = document.getElementById('btnInspectDefault');
  const datasetStatsGrid = document.getElementById('datasetStatsGrid');
  const statSource = document.getElementById('statSource');
  const statRows = document.getElementById('statRows');
  const statCols = document.getElementById('statCols');
  const statDist = document.getElementById('statDist');
  const previewContainer = document.getElementById('previewContainer');
  const datasetPreviewTable = document.getElementById('datasetPreviewTable');

  // DOM Elements - Admin Module 2 (Preprocess)
  const btnRunPreprocess = document.getElementById('btnRunPreprocess');
  const prepResults = document.getElementById('prepResults');
  const prepRawCount = document.getElementById('prepRawCount');
  const prepTransCount = document.getElementById('prepTransCount');
  const prepTrainCount = document.getElementById('prepTrainCount');
  const prepTestCount = document.getElementById('prepTestCount');

  // DOM Elements - Admin Module 3 (Train)
  const btnTrainModel = document.getElementById('btnTrainModel');
  const trainingOutput = document.getElementById('trainingOutput');
  const evalAlgoName = document.getElementById('evalAlgoName');
  const evalAcc = document.getElementById('evalAcc');
  const evalRoc = document.getElementById('evalRoc');
  const evalPrec = document.getElementById('evalPrec');
  const evalRec = document.getElementById('evalRec');
  const evalF1 = document.getElementById('evalF1');
  const cmTN = document.getElementById('cmTN');
  const cmFP = document.getElementById('cmFP');
  const cmFN = document.getElementById('cmFN');
  const cmTP = document.getElementById('cmTP');

  // DOM Elements - Admin Module 4 (Comparison)
  const btnLoadComparison = document.getElementById('btnLoadComparison');
  const comparisonContainer = document.getElementById('comparisonContainer');
  const comparisonBarsGrid = document.getElementById('comparisonBarsGrid');
  const comparisonTableBody = document.getElementById('comparisonTableBody');
  const comparisonConclusion = document.getElementById('comparisonConclusion');
  const btnApiConfig = document.getElementById('btnApiConfig');

  // =====================================================================
  // API BASE URL CONFIGURATION
  // Supports:
  // - Local Flask development (http://localhost:5000 or same-origin)
  // - Production Render backend URL (https://thyroid-xai.onrender.com)
  // - Dynamic runtime configuration via localStorage or window.API_BASE_URL
  // =====================================================================
  const PRODUCTION_RENDER_BACKEND = 'https://thyroid-xai.onrender.com';

  function getApiBaseUrl() {
    // 1. Explicit window override
    if (typeof window.API_BASE_URL === 'string' && window.API_BASE_URL.trim() !== '') {
      return window.API_BASE_URL.trim().replace(/\/+$/, '');
    }
    // 2. User-configured custom URL in modal
    const stored = localStorage.getItem('API_BASE_URL');
    if (stored && stored.trim() !== '') {
      return stored.trim().replace(/\/+$/, '');
    }
    // 3. Localhost development fallback
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      if (window.location.port === '5000') return ''; // same-origin relative when served directly by Flask backend
      return 'http://localhost:5000'; // local static dev server hitting local Flask
    }
    // 4. Default for production deployment on Vercel
    return PRODUCTION_RENDER_BACKEND;
  }

  let API_BASE_URL = getApiBaseUrl();

  // Cache for loaded preset samples
  let cachedSamples = {};

  // All 29 feature keys matching backend
  const NUMERICAL_KEYS = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'];
  const CATEGORICAL_KEYS = [
    'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured',
    'TBG_measured', 'referral_source'
  ];

  // Friendly human-readable feature labels
  const FEATURE_LABELS = {
    age: 'Patient Age',
    sex: 'Biological Sex',
    referral_source: 'Referral Source',
    TSH: 'Serum TSH (mIU/L)',
    T3: 'Total T3 (nmol/L)',
    TT4: 'Total T4 (nmol/L)',
    T4U: 'T4U Ratio',
    FTI: 'Free Thyroxine Index (FTI)',
    TBG: 'Thyroid-Binding Globulin (TBG)',
    TSH_measured: 'TSH Measured',
    T3_measured: 'T3 Measured',
    TT4_measured: 'TT4 Measured',
    T4U_measured: 'T4U Measured',
    FTI_measured: 'FTI Measured',
    TBG_measured: 'TBG Measured',
    on_thyroxine: 'On Thyroxine Therapy',
    query_on_thyroxine: 'Query on Thyroxine',
    on_antithyroid_medication: 'On Antithyroid Meds',
    sick: 'General Illness / Sick',
    pregnant: 'Pregnancy',
    thyroid_surgery: 'Prior Thyroid Surgery',
    I131_treatment: 'I131 Radiation Therapy',
    query_hypothyroid: 'Query Hypothyroid',
    query_hyperthyroid: 'Query Hyperthyroid',
    lithium: 'Lithium Therapy',
    goitre: 'Goitre Present',
    tumor: 'Thyroid Tumor',
    hypopituitary: 'Hypopituitary',
    psych: 'Psychiatric Symptoms'
  };

  // 1. Initial Health Check & Metadata
  async function initSystem() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Update System Status
      const dot = systemStatus ? systemStatus.querySelector('.status-dot') : null;
      if (dot) dot.className = 'status-dot online';
      if (statusText) statusText.textContent = 'API Connected';

      // Update Metrics Strip
      if (metaModel) metaModel.textContent = data.model_name;
      if (metaAccuracy) metaAccuracy.textContent = data.metrics.accuracy;
      if (metaRoc) metaRoc.textContent = data.metrics.roc_auc;
      if (metaPR) metaPR.textContent = `Precision ${data.metrics.precision} • Recall ${data.metrics.recall} • F1 ${data.metrics.f1_score}`;
      if (metaXai) metaXai.textContent = data.xai_engine;

      // Load Presets
      loadPresets();

    } catch (err) {
      console.error('System health check error:', err);
      const dot = systemStatus ? systemStatus.querySelector('.status-dot') : null;
      if (dot) dot.className = 'status-dot offline';
      if (statusText) statusText.textContent = 'API Offline';
    }
  }

  // 2. Fetch Sample Patient Presets
  async function loadPresets() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/samples`);
      if (!res.ok) return;
      const json = await res.json();
      if (json.status === 'success' && json.samples) {
        cachedSamples = json.samples;
      }
    } catch (err) {
      console.warn('Could not fetch sample profiles:', err);
    }
  }

  // 3. Populate Form with Data
  function populateForm(data) {
    if (!data) return;

    // Numerical & select inputs
    Object.keys(data).forEach(key => {
      const el = document.getElementById(key);
      if (!el) return;

      if (el.type === 'checkbox') {
        el.checked = (String(data[key]).toLowerCase() === 't');
      } else {
        el.value = data[key];
      }
    });

    // Provide visual feedback
    patientForm.classList.add('flash-highlight');
    setTimeout(() => patientForm.classList.remove('flash-highlight'), 500);
  }

  // Preset Button Listeners
  btnLoadPositive.addEventListener('click', () => {
    if (cachedSamples.positive_case) {
      populateForm(cachedSamples.positive_case.data);
    } else {
      // Fallback verified positive test values
      populateForm({
        age: 33, sex: 'F', referral_source: 'other',
        TSH: 6.6, T3: 1.9, TT4: 114, T4U: 0.93, FTI: 123, TBG: 26,
        TSH_measured: 't', T3_measured: 't', TT4_measured: 't', T4U_measured: 't', FTI_measured: 't', TBG_measured: 't',
        on_thyroxine: 'f', thyroid_surgery: 'f', sick: 'f', pregnant: 'f'
      });
    }
  });

  btnLoadNegative.addEventListener('click', () => {
    if (cachedSamples.negative_case) {
      populateForm(cachedSamples.negative_case.data);
    } else {
      // Fallback verified normal test values
      populateForm({
        age: 45, sex: 'F', referral_source: 'other',
        TSH: 1.4, T3: 2.1, TT4: 102, T4U: 0.98, FTI: 104, TBG: 22,
        TSH_measured: 't', T3_measured: 't', TT4_measured: 't', T4U_measured: 't', FTI_measured: 't', TBG_measured: 'f',
        on_thyroxine: 'f', thyroid_surgery: 'f', sick: 'f', pregnant: 'f'
      });
    }
  });

  btnResetForm.addEventListener('click', () => {
    patientForm.reset();
    emptyState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    resultContent.classList.add('hidden');
  });

  // 4. Gather Form Data into 29 Features Object
  function getFormData() {
    const data = {};

    // Numerical
    NUMERICAL_KEYS.forEach(key => {
      const el = document.getElementById(key);
      data[key] = (el && el.value !== '') ? parseFloat(el.value) : null;
    });

    // Categorical
    CATEGORICAL_KEYS.forEach(key => {
      const el = document.getElementById(key);
      if (!el) return;
      if (el.type === 'checkbox') {
        data[key] = el.checked ? 't' : 'f';
      } else {
        data[key] = el.value || 'other';
      }
    });

    return data;
  }

  // 5. Render Diagnosis Outcome Card
  function renderOutcome(result) {
    const isDisease = (result.prediction === 1);
    
    // Outcome Card Theme
    outcomeCard.className = isDisease ? 'outcome-card disease-detected' : 'outcome-card healthy-detected';
    outcomeBadge.className = isDisease ? 'outcome-badge disease' : 'outcome-badge healthy';
    outcomeBadge.textContent = isDisease ? 'THYROID DISEASE DETECTED' : 'NO DISEASE DETECTED';

    // Confidence & Progress Bar
    const conf = result.confidence_percent;
    confidenceValue.textContent = `${conf}%`;
    confidenceBar.style.width = `${conf}%`;
    confidenceBar.className = isDisease ? 'gauge-fill disease' : 'gauge-fill healthy';

    // Probabilities
    const healthyPct = (result.probability_healthy * 100).toFixed(1);
    const diseasePct = (result.probability_disease * 100).toFixed(1);
    probHealthy.textContent = `${healthyPct}%`;
    probDisease.textContent = `${diseasePct}%`;
  }

  // 6. Render DiCE Counterfactual Explanations
  function renderCounterfactuals(xaiData) {
    cfListContainer.innerHTML = '';
    xaiContainer.classList.remove('hidden');

    const origPred = xaiData.original_prediction.prediction;
    const desired = xaiData.desired_class;
    const isOrigDisease = (origPred === 1);

    xaiSubtitle.textContent = isOrigDisease
      ? 'What minimal marker changes would cause the model to classify this patient as Healthy?'
      : 'What feature changes would cross the decision boundary into Thyroid Disease?';

    const cfs = xaiData.counterfactuals || [];
    if (cfs.length === 0) {
      cfListContainer.innerHTML = `
        <div class="cf-card">
          <p style="color: var(--text-muted); font-size: 0.85rem;">
            No direct counterfactuals found within the standard search bounds for this specific profile.
          </p>
        </div>
      `;
      return;
    }

    cfs.forEach((cf, idx) => {
      const isCfHealthy = (cf.prediction === 0);
      const card = document.createElement('div');
      card.className = 'cf-card';

      // Header
      const header = document.createElement('div');
      header.className = 'cf-card-header';
      header.innerHTML = `
        <span class="cf-badge">Counterfactual Explanation #${idx + 1}</span>
        <span class="cf-outcome-pill ${isCfHealthy ? 'healthy' : 'disease'}">
          Model Prediction: ${cf.label} (${cf.confidence_percent}% confidence)
        </span>
      `;
      card.appendChild(header);

      // Feature changes table
      const changes = cf.changed_features || [];
      if (changes.length > 0) {
        let tableHtml = `
          <table class="diff-table">
            <thead>
              <tr>
                <th>Clinical Feature</th>
                <th>Patient's Value</th>
                <th>Counterfactual Target</th>
                <th>Decision Shift (Δ)</th>
              </tr>
            </thead>
            <tbody>
        `;

        changes.forEach(c => {
          const label = FEATURE_LABELS[c.feature] || c.feature;
          const diffStr = (typeof c.difference === 'number')
            ? (c.difference > 0 ? `+${c.difference}` : `${c.difference}`)
            : `${c.difference}`;

          tableHtml += `
            <tr>
              <td class="feature-name">${label}</td>
              <td class="val-orig">${c.original_value}</td>
              <td class="val-cf">${c.counterfactual_value}</td>
              <td><span class="delta-badge">${diffStr}</span></td>
            </tr>
          `;
        });

        tableHtml += `</tbody></table>`;
        
        // Narrative summary
        const topChange = changes[0];
        const topLabel = FEATURE_LABELS[topChange.feature] || topChange.feature;
        const narrativeText = isOrigDisease
          ? `The Random Forest model flips to <strong>Normal</strong> if <strong>${topLabel}</strong> adjusts from ${topChange.original_value} to ${topChange.counterfactual_value}.`
          : `The model alters prediction to <strong>Thyroid Disease</strong> if <strong>${topLabel}</strong> shifts from ${topChange.original_value} to ${topChange.counterfactual_value}.`;

        tableHtml += `
          <div class="cf-summary-narrative">
            💡 <strong>Algorithmic Boundary Shift:</strong> ${narrativeText}
          </div>
        `;

        const diffContainer = document.createElement('div');
        diffContainer.innerHTML = tableHtml;
        card.appendChild(diffContainer);
      } else {
        card.innerHTML += `<p style="font-size: 0.82rem; color: var(--text-muted);">Prediction shifted with minimal internal feature boundaries.</p>`;
      }

      cfListContainer.appendChild(card);
    });
  }

  // 7. Handle Full Diagnose + Counterfactual XAI
  patientForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = getFormData();
    emptyState.classList.add('hidden');
    resultContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingTitle.textContent = 'Analyzing Patient Profile...';
    loadingMsg.textContent = 'Random Forest evaluating 29 features & DiCE computing counterfactuals...';

    try {
      const res = await fetch(`${API_BASE_URL}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: formData,
          total_cfs: 2
        })
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.message || `Server Error ${res.status}`);
      }

      const json = await res.json();
      const xaiData = json.data;

      // Render Outcome
      renderOutcome({
        prediction: xaiData.original_prediction.prediction,
        confidence_percent: xaiData.original_prediction.confidence_percent,
        probability_healthy: xaiData.original_prediction.probability_healthy,
        probability_disease: xaiData.original_prediction.probability_disease
      });

      // Render Counterfactuals
      renderCounterfactuals(xaiData);

      // Display Content
      loadingState.classList.add('hidden');
      resultContent.classList.remove('hidden');
      const resultsSection = document.getElementById('resultsSection');
      if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

    } catch (err) {
      console.error('Diagnosis & Explain error:', err);
      loadingState.classList.add('hidden');
      emptyState.classList.remove('hidden');
      alert(`Error during evaluation: ${err.message}`);
    }
  });

  // 8. Handle Quick Diagnosis Only
  btnQuickPredict.addEventListener('click', async () => {
    const formData = getFormData();
    emptyState.classList.add('hidden');
    resultContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingTitle.textContent = 'Running Fast Prediction...';
    loadingMsg.textContent = 'Evaluating Random Forest ensemble without counterfactual generation...';

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.message || `Server Error ${res.status}`);
      }

      const json = await res.json();
      const predData = json.data;

      renderOutcome(predData);
      xaiContainer.classList.add('hidden'); // Hide counterfactuals for quick predict

      loadingState.classList.add('hidden');
      resultContent.classList.remove('hidden');
      const resultsSection = document.getElementById('resultsSection');
      if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

    } catch (err) {
      console.error('Quick predict error:', err);
      loadingState.classList.add('hidden');
      emptyState.classList.remove('hidden');
      alert(`Error during prediction: ${err.message}`);
    }
  });

  // =====================================================================
  // TAB NAVIGATION
  // =====================================================================
  tabClinical.addEventListener('click', () => {
    tabClinical.classList.add('active');
    if (navXai) navXai.classList.remove('active');
    tabAdmin.classList.remove('active');
    clinicalView.classList.remove('hidden');
    adminView.classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  if (navXai) {
    navXai.addEventListener('click', () => {
      tabClinical.classList.remove('active');
      navXai.classList.add('active');
      tabAdmin.classList.remove('active');
      clinicalView.classList.remove('hidden');
      adminView.classList.add('hidden');
      const target = document.getElementById('xaiContainer') || document.getElementById('resultsSection');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  }

  tabAdmin.addEventListener('click', () => {
    tabAdmin.classList.add('active');
    tabClinical.classList.remove('active');
    if (navXai) navXai.classList.remove('active');
    adminView.classList.remove('hidden');
    clinicalView.classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Automatically inspect current dataset if not loaded yet
    if (statRows.textContent === '--') {
      inspectDataset();
    }
  });

  // =====================================================================
  // ADMIN MODULE 1: UPLOAD & INSPECT DATASET
  // =====================================================================
  datasetFileInput.addEventListener('change', () => {
    if (datasetFileInput.files && datasetFileInput.files[0]) {
      uploadLabelText.textContent = `Selected: ${datasetFileInput.files[0].name}`;
    }
  });

  async function inspectDataset(file = null) {
    try {
      let res;
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        res = await fetch(`${API_BASE_URL}/api/admin/upload-dataset`, {
          method: 'POST',
          body: formData
        });
      } else {
        res = await fetch(`${API_BASE_URL}/api/admin/dataset-info`);
      }

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || 'Failed to inspect dataset');
      }

      const info = await res.json();
      renderDatasetInfo(info);

    } catch (err) {
      console.error('Dataset inspection error:', err);
      alert(`Error: ${err.message}`);
    }
  }

  function renderDatasetInfo(info) {
    datasetStatsGrid.classList.remove('hidden');
    previewContainer.classList.remove('hidden');

    statSource.textContent = info.source;
    statRows.textContent = info.total_rows.toLocaleString();
    statCols.textContent = info.total_columns;

    if (info.target_distribution) {
      const h = info.target_distribution.class_0_healthy || 0;
      const d = info.target_distribution.class_1_disease || 0;
      statDist.textContent = `0 (Normal): ${h} | 1 (Disease): ${d}`;
    } else {
      statDist.textContent = 'N/A';
    }

    // Render Preview Table
    if (info.preview && info.preview.length > 0) {
      const cols = info.columns || Object.keys(info.preview[0]);
      let tableHtml = '<table><thead><tr>';
      cols.forEach(c => { tableHtml += `<th>${c}</th>`; });
      tableHtml += '</tr></thead><tbody>';

      info.preview.forEach(row => {
        tableHtml += '<tr>';
        cols.forEach(c => {
          tableHtml += `<td>${row[c] !== null && row[c] !== undefined ? row[c] : ''}</td>`;
        });
        tableHtml += '</tr>';
      });

      tableHtml += '</tbody></table>';
      datasetPreviewTable.innerHTML = tableHtml;
    }
  }

  btnUploadDataset.addEventListener('click', () => {
    if (datasetFileInput.files && datasetFileInput.files[0]) {
      inspectDataset(datasetFileInput.files[0]);
    } else {
      alert('Please choose a CSV file first, or click "Inspect Active Cleaned Dataset".');
    }
  });

  btnInspectDefault.addEventListener('click', () => {
    inspectDataset();
  });

  // =====================================================================
  // ADMIN MODULE 2: PREPROCESS DATASET
  // =====================================================================
  btnRunPreprocess.addEventListener('click', async () => {
    btnRunPreprocess.disabled = true;
    btnRunPreprocess.textContent = 'Processing Pipeline...';

    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/preprocess`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || 'Preprocessing failed');
      }

      const data = await res.json();
      prepResults.classList.remove('hidden');

      prepRawCount.textContent = data.raw_feature_count;
      prepTransCount.textContent = data.transformed_feature_count;
      prepTrainCount.textContent = data.train_samples.toLocaleString();
      prepTestCount.textContent = data.test_samples.toLocaleString();

    } catch (err) {
      console.error('Preprocessing error:', err);
      alert(`Preprocessing error: ${err.message}`);
    } finally {
      btnRunPreprocess.disabled = false;
      btnRunPreprocess.textContent = 'Execute Preprocessing Pipeline';
    }
  });

  // =====================================================================
  // ADMIN MODULE 3: APPLY ALGORITHM
  // =====================================================================
  btnTrainModel.addEventListener('click', async () => {
    const selectedAlgo = document.querySelector('input[name="adminAlgo"]:checked').value;
    btnTrainModel.disabled = true;
    btnTrainModel.textContent = 'Training & Evaluating in Memory...';

    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: selectedAlgo })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || 'Training failed');
      }

      const json = await res.json();
      const result = json.data;

      trainingOutput.classList.remove('hidden');
      evalAlgoName.textContent = result.algorithm;
      evalAcc.textContent = `${result.metrics.accuracy}%`;
      evalRoc.textContent = result.metrics.roc_auc;
      evalPrec.textContent = `${result.metrics.precision}%`;
      evalRec.textContent = `${result.metrics.recall}%`;
      evalF1.textContent = `${result.metrics.f1_score}%`;

      // Confusion Matrix
      const cm = result.confusion_matrix;
      cmTN.textContent = cm.true_negative;
      cmFP.textContent = cm.false_positive;
      cmFN.textContent = cm.false_negative;
      cmTP.textContent = cm.true_positive;

    } catch (err) {
      console.error('Training error:', err);
      alert(`Training error: ${err.message}`);
    } finally {
      btnTrainModel.disabled = false;
      btnTrainModel.textContent = 'Train Model & Calculate Metrics';
    }
  });

  // =====================================================================
  // ADMIN MODULE 4: ALGORITHM COMPARISON GRAPH
  // =====================================================================
  btnLoadComparison.addEventListener('click', async () => {
    btnLoadComparison.disabled = true;
    btnLoadComparison.textContent = 'Calculating Comparison...';

    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/comparison`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || 'Comparison failed');
      }

      const json = await res.json();
      renderComparison(json.data);

    } catch (err) {
      console.error('Comparison error:', err);
      alert(`Comparison error: ${err.message}`);
    } finally {
      btnLoadComparison.disabled = false;
      btnLoadComparison.textContent = 'Generate / Refresh Comparison Graph';
    }
  });

  function renderComparison(data) {
    comparisonContainer.classList.remove('hidden');

    const lr = data.algorithms.find(a => a.name === 'Logistic Regression');
    const rf = data.algorithms.find(a => a.name === 'Random Forest');

    if (!lr || !rf) return;

    // 1. Render Dual Horizontal Bars
    const metricsToDisplay = [
      { name: 'Model Accuracy', lrVal: lr.accuracy, rfVal: rf.accuracy, unit: '%' },
      { name: 'ROC-AUC Score', lrVal: lr.roc_auc, rfVal: rf.roc_auc, unit: '% (Normalized)' },
      { name: 'Precision', lrVal: lr.precision, rfVal: rf.precision, unit: '%' },
      { name: 'Recall / Sensitivity', lrVal: lr.recall, rfVal: rf.recall, unit: '%' },
      { name: 'F1-Score', lrVal: lr.f1_score, rfVal: rf.f1_score, unit: '%' }
    ];

    let barsHtml = '';
    metricsToDisplay.forEach(m => {
      barsHtml += `
        <div class="comp-metric-row">
          <div class="comp-metric-label-row">
            <span><strong>${m.name}</strong></span>
            <span>LR: ${m.lrVal}% vs <strong style="color: var(--primary);">RF: ${m.rfVal}%</strong></span>
          </div>
          <div class="comp-bar-pair">
            <div class="comp-bar lr-bar" style="width: ${Math.max(12, m.lrVal)}%;">Log. Regression: ${m.lrVal}%</div>
            <div class="comp-bar rf-bar" style="width: ${Math.max(12, m.rfVal)}%;">Random Forest: ${m.rfVal}%</div>
          </div>
        </div>
      `;
    });
    comparisonBarsGrid.innerHTML = barsHtml;

    // 2. Render Table
    let tableRows = '';
    metricsToDisplay.forEach(m => {
      const diff = (m.rfVal - m.lrVal).toFixed(2);
      tableRows += `
        <tr>
          <td><strong>${m.name}</strong></td>
          <td>${m.lrVal}%</td>
          <td><strong>${m.rfVal}%</strong></td>
          <td><span class="diff-positive">+${diff}%</span></td>
        </tr>
      `;
    });
    comparisonTableBody.innerHTML = tableRows;

    // 3. Render Analytical Conclusion
    comparisonConclusion.innerHTML = `
      <strong>💡 Viva Analytical Conclusion:</strong><br>
      ${data.conclusion}
    `;
  }

  // =====================================================================
  // RUNTIME API BASE URL CONFIGURATION DIALOG
  // =====================================================================
  if (btnApiConfig) {
    btnApiConfig.addEventListener('click', () => {
      const current = localStorage.getItem('API_BASE_URL') || API_BASE_URL || PRODUCTION_RENDER_BACKEND;
      const input = prompt(
        'Configure Backend API URL for Render / Cloud Deployment:\n\n' +
        'Default Render API: https://thyroid-xai.onrender.com\n' +
        'Localhost: http://localhost:5000\n\n' +
        'Current API URL:',
        current
      );
      if (input !== null) {
        const trimmed = input.trim().replace(/\/+$/, '');
        if (trimmed) {
          localStorage.setItem('API_BASE_URL', trimmed);
        } else {
          localStorage.removeItem('API_BASE_URL');
        }
        API_BASE_URL = getApiBaseUrl();
        initSystem();
        alert(`API Base URL updated to: ${API_BASE_URL || '(same-origin relative)'}`);
      }
    });
  }

  // Run initial setup
  initSystem();
});
