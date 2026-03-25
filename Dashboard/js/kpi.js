const UIManager = {
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
    },

    formatNumber(value) {
        if (value >= 1e9) {
            return (value / 1e9).toFixed(2) + 'B';
        }
        if (value >= 1e6) {
            return (value / 1e6).toFixed(2) + 'M';
        }
        return new Intl.NumberFormat('en-US').format(value);
    },

    renderKPIs(dataStr = 'all', startDate = null, endDate = null) {
        const container = document.getElementById('kpi-container');
        const data = DataService.getFilteredHistory(dataStr, startDate, endDate);
        if (!data || data.length === 0) return;

        const latest = data[data.length - 1];
        const previous = data[data.length - 2];
        const latestClose = latest.Close;
        const priceChange = latestClose - (previous ? previous.Close : latestClose);
        const percentChange = previous ? (priceChange / previous.Close) * 100 : 0;
        
        const closes = data.map(d => d.Close);
        const high = Math.max(...closes);
        const low = Math.min(...closes);
        
        const avgVolume = data.reduce((sum, d) => sum + d.Volume, 0) / data.length;

        // Market cap from info string, but we can display the current price
        
        container.innerHTML = `
            <div class="kpi-card">
                <span class="kpi-title">Latest Close Price</span>
                <span class="kpi-value">${this.formatCurrency(latestClose)}</span>
                <span class="kpi-subtext ${priceChange >= 0 ? 'positive' : 'negative'}">
                    <i class="fa-solid fa-${priceChange >= 0 ? 'arrow-trend-up' : 'arrow-trend-down'}"></i>
                    ${priceChange >= 0 ? '+' : ''}${this.formatCurrency(priceChange)} (${percentChange.toFixed(2)}%)
                </span>
            </div>
            
            <div class="kpi-card">
                <span class="kpi-title">Period High</span>
                <span class="kpi-value">${this.formatCurrency(high)}</span>
                <span class="kpi-subtext neutral"><i class="fa-solid fa-arrow-up"></i> Highest in range</span>
            </div>
            
            <div class="kpi-card">
                <span class="kpi-title">Period Low</span>
                <span class="kpi-value">${this.formatCurrency(low)}</span>
                <span class="kpi-subtext neutral"><i class="fa-solid fa-arrow-down"></i> Lowest in range</span>
            </div>
            
            <div class="kpi-card">
                <span class="kpi-title">Average Volume</span>
                <span class="kpi-value">${this.formatNumber(avgVolume)}</span>
                <span class="kpi-subtext neutral"><i class="fa-solid fa-chart-bar"></i> Avg shares traded</span>
            </div>
        `;
    },

    renderInfoPanel() {
        const info = window.appData.info;
        if (!info) return;

        // Map info data to the new Project Info grid IDs
        const mappings = {
            'pf-mcap': this.formatNumber(info.marketCap) || 'N/A',
            'pf-pe': info.trailingPE || 'N/A',
            'pf-fpe': info.forwardPE || 'N/A',
            'pf-payout': (info.payoutRatio * 100).toFixed(2) + '%' || 'N/A',
            'pf-sector': info.sector || 'N/A',
            'pf-industry': info.industry || 'N/A',
            'pf-employees': this.formatNumber(info.fullTimeEmployees) || 'N/A'
        };

        for (const [id, value] of Object.entries(mappings)) {
            const el = document.getElementById(id);
            if (el) el.innerText = value;
        }
    },

    renderMLMetrics(stratFactor, marketFactor) {
        const sEl = document.getElementById('strat-factor');
        const mEl = document.getElementById('market-factor');
        if (sEl) sEl.innerText = stratFactor.toFixed(2) + 'x';
        if (mEl) mEl.innerText = marketFactor.toFixed(2) + 'x';
    }
};
