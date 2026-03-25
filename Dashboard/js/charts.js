const ChartManager = {
    getThemeColors() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        return {
            bg: 'transparent',
            text: isDark ? '#ffffff' : '#1e1e24',
            grid: isDark ? '#2a2a35' : '#e1e3ea',
            lineColor: '#F40009',
            ma20: '#00b4d8',
            ma50: '#ffb703',
            volUp: isDark ? '#00c853' : '#00b047',
            volDown: isDark ? '#f40009' : '#e00000',
            histColor: 'rgba(244, 0, 9, 0.6)'
        };
    },

    commonLayout() {
        const colors = this.getThemeColors();
        return {
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            font: { family: 'Inter, sans-serif', color: colors.text },
            margin: { t: 10, r: 10, l: 50, b: 30 },
            xaxis: {
                gridcolor: colors.grid,
                zerolinecolor: colors.grid
            },
            yaxis: {
                gridcolor: colors.grid,
                zerolinecolor: colors.grid
            }
        };
    },

    renderPriceChart(rangeStr, showMA, startDate = null, endDate = null) {
        const data = DataService.getFilteredHistory(rangeStr, startDate, endDate);
        const colors = this.getThemeColors();
        
        const dates = data.map(d => d.Date);
        const closes = data.map(d => d.Close);
        
        const traces = [
            {
                x: dates,
                y: closes,
                type: 'scatter',
                mode: 'lines',
                name: 'Close Price',
                line: { color: colors.lineColor, width: 2 }
            }
        ];

        if (showMA) {
            const ma20 = DataService.calculateMovingAverage(data, 20);
            const ma50 = DataService.calculateMovingAverage(data, 50);
            
            traces.push({
                x: dates,
                y: ma20,
                type: 'scatter',
                mode: 'lines',
                name: '20-Day MA',
                line: { color: colors.ma20, width: 1.5, dash: 'dot' }
            });
            
            traces.push({
                x: dates,
                y: ma50,
                type: 'scatter',
                mode: 'lines',
                name: '50-Day MA',
                line: { color: colors.ma50, width: 1.5, dash: 'dash' }
            });
        }

        const layout = {
            ...this.commonLayout(),
            showlegend: true,
            legend: { orientation: 'h', y: 1.1 },
            hovermode: 'x unified'
        };

        Plotly.newPlot('price-chart', traces, layout, { responsive: true, displayModeBar: false });
        
        // Update Trend Indicator Badge
        const latest = closes[closes.length - 1];
        const first = closes[0];
        const trendBadge = document.getElementById('trend-indicator');
        if (latest >= first) {
            trendBadge.className = 'trend-badge';
            trendBadge.innerHTML = '<i class="fa-solid fa-arrow-trend-up"></i> Uptrend';
        } else {
            trendBadge.className = 'trend-badge down';
            trendBadge.innerHTML = '<i class="fa-solid fa-arrow-trend-down"></i> Downtrend';
        }
    },

    renderVolumeChart(rangeStr, startDate = null, endDate = null) {
        const data = DataService.getFilteredHistory(rangeStr, startDate, endDate);
        const colors = this.getThemeColors();
        
        const dates = data.map(d => d.Date);
        const volumes = data.map(d => d.Volume);
        
        // Color volume bars based on close price direction
        const barColors = data.map((d, i) => {
            if (i === 0) return colors.volUp;
            return d.Close >= data[i-1].Close ? colors.volUp : colors.volDown;
        });

        // Calculate average volume to highlight spikes
        const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
        const spikeThreshold = avgVolume * 2;
        
        // Highlight spikes with brighter color
        const finalColors = barColors.map((c, i) => {
            if (volumes[i] > spikeThreshold) return '#ffea00'; // Highlight color
            return c;
        });

        const trace = {
            x: dates,
            y: volumes,
            type: 'bar',
            name: 'Volume',
            marker: { color: finalColors }
        };

        const layout = {
            ...this.commonLayout(),
            margin: { t: 10, r: 10, l: 50, b: 30 },
        };

        Plotly.newPlot('volume-chart', [trace], layout, { responsive: true, displayModeBar: false });
    },

    renderDistributionChart(rangeStr, startDate = null, endDate = null) {
        const data = DataService.getFilteredHistory(rangeStr, startDate, endDate);
        const colors = this.getThemeColors();
        const closes = data.map(d => d.Close);

        const trace = {
            x: closes,
            type: 'histogram',
            marker: { color: colors.histColor, line: { color: colors.lineColor, width: 1 } },
            opacity: 0.7
        };

        const layout = {
            ...this.commonLayout(),
            xaxis: { ...this.commonLayout().xaxis, title: 'Price (USD)' },
            yaxis: { ...this.commonLayout().yaxis, title: 'Frequency' },
            bargap: 0.05
        };

        Plotly.newPlot('distribution-chart', [trace], layout, { responsive: true, displayModeBar: false });
    },

    renderMLPriceChart(mlData) {
        if (!mlData) return;
        const colors = this.getThemeColors();

        const traceActual = {
            x: mlData.dates,
            y: mlData.actualClose,
            type: 'scatter',
            mode: 'lines',
            name: 'Actual Close',
            line: { color: colors.text, width: 2, dash: 'solid' },
            opacity: 0.6
        };

        const tracePredicted = {
            x: mlData.dates,
            y: mlData.predictedClose,
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted Close',
            line: { color: '#00b4d8', width: 1.5, dash: 'solid' }
        };

        const layout = {
            ...this.commonLayout(),
            showlegend: true,
            legend: { orientation: 'h', y: 1.1 },
            hovermode: 'x unified',
            yaxis: { ...this.commonLayout().yaxis, title: 'Price ($)' }
        };

        Plotly.newPlot('ml-price-chart', [traceActual, tracePredicted], layout, { responsive: true, displayModeBar: false });
    },

    renderMLReturnsChart(mlData) {
        if (!mlData) return;
        const colors = this.getThemeColors();

        const traceMarket = {
            x: mlData.dates,
            y: mlData.marketReturns,
            type: 'scatter',
            mode: 'lines',
            name: 'Market Return',
            line: { color: colors.text, width: 2 },
            opacity: 0.6
        };

        const traceStrat = {
            x: mlData.dates,
            y: mlData.strategyReturns,
            type: 'scatter',
            mode: 'lines',
            name: 'Strategy Return',
            line: { color: '#00c853', width: 2 }
        };

        const layout = {
            ...this.commonLayout(),
            showlegend: true,
            legend: { orientation: 'h', y: 1.1 },
            hovermode: 'x unified',
            yaxis: { ...this.commonLayout().yaxis, title: 'Cumulative Return Factor' }
        };

        Plotly.newPlot('ml-returns-chart', [traceMarket, traceStrat], layout, { responsive: true, displayModeBar: false });
    },

    renderAll(rangeStr, showMA, startDate = null, endDate = null) {
        this.renderPriceChart(rangeStr, showMA, startDate, endDate);
        this.renderVolumeChart(rangeStr, startDate, endDate);
        this.renderDistributionChart(rangeStr, startDate, endDate);
    }
};
