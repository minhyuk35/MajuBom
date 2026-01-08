(() => {
  const animateCards = () => {
    const items = document.querySelectorAll('[data-animate]');
    items.forEach((item, index) => {
      setTimeout(() => item.classList.add('appear'), index * 120);
    });
  };

  const renderRegionChart = (id, data) => {
    const ctx = document.getElementById(id);
    if (!ctx || !window.Chart) return;
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: '안정',
            data: data.safe,
            backgroundColor: '#18a16c',
            borderRadius: 8,
          },
          {
            label: '주의',
            data: data.caution,
            backgroundColor: '#f0b429',
            borderRadius: 8,
          },
          {
            label: '위험',
            data: data.danger,
            backgroundColor: '#ff4d5a',
            borderRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: { stacked: true, grid: { display: false } },
          y: { stacked: true, ticks: { stepSize: 2 } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: '#16141f' },
        },
      },
    });
  };

  const renderMissionChart = (id, data) => {
    const ctx = document.getElementById(id);
    if (!ctx || !window.Chart) return;
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['완료', '미완료'],
        datasets: [
          {
            data: [data.completed, data.remaining],
            backgroundColor: ['#ff7a45', '#f2e8db'],
            borderWidth: 0,
          },
        ],
      },
      options: {
        cutout: '70%',
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: '#16141f' },
        },
      },
    });
  };

  const renderActivityCharts = () => {
    if (!window.Chart) return;
    document.querySelectorAll('canvas[id^="activity-"]').forEach((canvas) => {
      const scriptId = canvas.id.replace('activity-', 'activity-data-');
      const dataEl = document.getElementById(scriptId);
      if (!dataEl) return;
      const data = JSON.parse(dataEl.textContent);
      new Chart(canvas, {
        type: 'line',
        data: {
          labels: data.labels,
          datasets: [
            {
              data: data.values,
              borderColor: '#2d2a3a',
              backgroundColor: 'rgba(45, 42, 58, 0.12)',
              fill: true,
              tension: 0.35,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: { display: false },
          },
        },
      });
    });
  };

  const renderSkeletons = () => {
    document.querySelectorAll('canvas[id^="skeleton-"]').forEach((canvas) => {
      const scriptId = canvas.id.replace('skeleton-', 'skeleton-data-');
      const dataEl = document.getElementById(scriptId);
      if (!dataEl) return;
      const points = JSON.parse(dataEl.textContent);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#16141f';
      ctx.lineWidth = 2;
      ctx.fillStyle = '#ff7a45';
      const connect = [
        [0, 1], [1, 2], [2, 3],
        [1, 4], [4, 5], [5, 6],
        [1, 7], [7, 8], [8, 9],
        [7, 10], [10, 11], [11, 12],
      ];
      connect.forEach(([a, b]) => {
        if (!points[a] || !points[b]) return;
        ctx.beginPath();
        ctx.moveTo(points[a].x, points[a].y);
        ctx.lineTo(points[b].x, points[b].y);
        ctx.stroke();
      });
      points.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  };

  window.renderRegionChart = renderRegionChart;
  window.renderMissionChart = renderMissionChart;
  window.renderActivityCharts = renderActivityCharts;
  window.renderSkeletons = renderSkeletons;

  document.addEventListener('DOMContentLoaded', () => {
    animateCards();
  });
})();
