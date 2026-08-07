/*
 * EXAI-ResumeIntel: Landing page interactions
 * Author: Mithin Sagar S
 * GitHub: https://github.com/mithinsagar
 * Institution: Vellore Institute of Technology (VIT)
 */

(function () {
    'use strict';

    // Smooth-scroll for in-page anchors
    document.querySelectorAll('a[href^="#"]').forEach((link) => {
        link.addEventListener('click', (event) => {
            const targetId = link.getAttribute('href').slice(1);
            if (!targetId) return;
            const target = document.getElementById(targetId);
            if (!target) return;
            event.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    // Detect if the API is reachable (best-effort)
    const API_BASE = window.EXAI_API_BASE || 'http://localhost:8765';
    const statusEl = document.getElementById('api-status');
    if (statusEl) {
        fetch(`${API_BASE}/health`, { method: 'GET' })
            .then((resp) => resp.ok ? resp.json() : Promise.reject())
            .then((data) => {
                statusEl.textContent = `API online (v${data.version})`;
                statusEl.classList.add('online');
            })
            .catch(() => {
                statusEl.textContent = 'API offline';
                statusEl.classList.add('offline');
            });
    }
})();
