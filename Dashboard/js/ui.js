/**
 * UI Service for handling tab navigation, views, and custom slider logic.
 */
class UIService {
    static init() {
        this.initTabs();
    }

    static initTabs() {
        const navLinks = document.querySelectorAll('.nav-link');
        const views = document.querySelectorAll('.view-section');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Remove active class from all links
                navLinks.forEach(l => l.classList.remove('active'));
                
                // Add active class to clicked link
                link.classList.add('active');

                // Hide all views
                views.forEach(view => {
                    view.classList.remove('active');
                    view.classList.add('hidden');
                });

                // Show target view
                const targetViewId = link.getAttribute('data-view');
                const targetView = document.getElementById(targetViewId);
                if (targetView) {
                    targetView.classList.remove('hidden');
                    targetView.classList.add('active');
                }

                // If switching to ML view, make sure it renders
                if (targetViewId === 'ml-view' && window.MLService) {
                    // Trigger a resize event to ensure Plotly charts fit the container properly if they were hidden
                    window.dispatchEvent(new Event('resize'));
                }
            });
        });
    }

    static initDateSlider(minDateStr, maxDateStr, onChangeCallback) {
        const sliderMin = document.getElementById('date-slider-min');
        const sliderMax = document.getElementById('date-slider-max');
        const sliderFill = document.getElementById('slider-fill');
        const minLabel = document.getElementById('date-min-label');
        const maxLabel = document.getElementById('date-max-label');

        // Parse dates
        const minDate = new Date(minDateStr).getTime();
        const maxDate = new Date(maxDateStr).getTime();
        const datesSpan = maxDate - minDate;

        const updateUI = () => {
            let val1 = parseInt(sliderMin.value);
            let val2 = parseInt(sliderMax.value);

            // Prevent sliders from crossing
            if (val1 > val2) {
                const tmp = val1;
                val1 = val2;
                val2 = tmp;
            }

            // Update fill track
            sliderFill.style.left = `${val1}%`;
            sliderFill.style.width = `${val2 - val1}%`;

            // Calculate dates
            const curMinDate = new Date(minDate + (val1 / 100) * datesSpan);
            const curMaxDate = new Date(minDate + (val2 / 100) * datesSpan);

            const formatStr = (d) => d.toISOString().split('T')[0];

            minLabel.innerText = formatStr(curMinDate);
            maxLabel.innerText = formatStr(curMaxDate);

            // Call the chart update callback
            if (onChangeCallback) {
                onChangeCallback(formatStr(curMinDate), formatStr(curMaxDate));
            }
        };

        sliderMin.addEventListener('input', updateUI);
        sliderMax.addEventListener('input', updateUI);

        // Initial setup
        sliderMin.value = 0;
        sliderMax.value = 100;
        updateUI();
    }
}

window.UIService = UIService;
