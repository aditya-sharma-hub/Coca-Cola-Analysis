/**
 * Main Application Orchestrator
 */
document.addEventListener('DOMContentLoaded', async () => {
    const kpiContainer = document.getElementById('kpi-container');
    
    // 1. Initialize data
    try {
        await DataService.init();
        window.appData = {
            history: DataService.historyData,
            info: DataService.infoData
        };
    } catch (e) {
        console.error("Failed to parse data:", e);
        kpiContainer.innerHTML = '<div style="color:var(--negative); padding: 20px;">Failed to load data. Ensure CSV strings were generated correctly.</div>';
        return;
    }

    // 2. Initialize ML & UI Services
    MLService.init();
    UIService.init();

    // 3. UI Elements
    const dateRangeSelect = document.getElementById('date-range');
    const maToggleInput = document.getElementById('toggle-ma');
    
    let currentStartDate = '1962-01-02';
    let currentEndDate = '2022-03-11';
    let currentRange = 'all';

    function updateAllViews() {
        // Dashboard View
        if (kpiContainer) kpiContainer.classList.remove('loading');
        
        const showMA = maToggleInput.checked;
        
        // Render Dashboard with custom dates from slider
        UIManager.renderKPIs(currentRange, currentStartDate, currentEndDate);
        ChartManager.renderAll(currentRange, showMA, currentStartDate, currentEndDate);
        
        // ML Prediction View
        const mlData = MLService.getFilteredResults(currentStartDate, currentEndDate);
        if (mlData) {
            ChartManager.renderMLPriceChart(mlData);
            ChartManager.renderMLReturnsChart(mlData);
            UIManager.renderMLMetrics(mlData.stratFinalFactor, mlData.marketFinalFactor);
            
            // Update subtitle dates
            const subTitle = document.getElementById('ml-subtitle-dates');
            if (subTitle) subTitle.innerText = `Visualizing simulated ML performance and strategy over the period: ${currentStartDate} to ${currentEndDate}.`;
            
            const apHead = document.getElementById('actual-pred-head');
            if (apHead) apHead.innerText = `Actual vs. Predicted Close Price (${currentStartDate} to ${currentEndDate})`;
            
            const spHead = document.getElementById('strat-perf-head');
            if (spHead) spHead.innerText = `Cumulative Returns: Custom Strategy vs. Market (${currentStartDate} to ${currentEndDate})`;
        }

        // Project Info View
        UIManager.renderInfoPanel();
    }

    // 4. Setup Range Slider Logic (GLOBAL)
    UIService.initDateSlider('1962-01-02', '2022-03-11', (min, max) => {
        currentStartDate = min;
        currentEndDate = max;
        // Optimization: if user moved slider, the "Presets" like '5y' are no longer active, so we use 'custom'
        currentRange = 'custom'; 
        updateAllViews();
    });

    // 5. Setup Listeners
    if (dateRangeSelect) {
        dateRangeSelect.addEventListener('change', () => {
            currentRange = dateRangeSelect.value;
            // When selecting a preset, the slider should ideally reset, but for now we prioritize the preset if changed
            updateAllViews();
        });
    }

    maToggleInput.addEventListener('change', () => {
        const range = dateRangeSelect ? dateRangeSelect.value : 'all';
        ChartManager.renderPriceChart(range, maToggleInput.checked, currentStartDate, currentEndDate);
    });

    // 6. Theme Toggle
    const themeBtn = document.getElementById('theme-toggle');
    const htmlEl = document.documentElement;
    
    themeBtn.addEventListener('click', () => {
        const currentTheme = htmlEl.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        htmlEl.setAttribute('data-theme', newTheme);
        
        setTimeout(() => updateAllViews(), 100);
    });

    // 7. Initial Call
    updateAllViews();

    // 8. Resize Logic
    window.addEventListener('resize', () => {
        const charts = ['price-chart', 'volume-chart', 'distribution-chart', 'ml-price-chart', 'ml-returns-chart'];
        charts.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.innerHTML !== '') Plotly.Plots.resize(id);
        });
    });
});
