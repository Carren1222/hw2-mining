(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var accent3 = style.getPropertyValue('--accent3').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var NAMES = ['年轻潜力客群', '富裕稳健客群', '主力中产客群'];
  var COLORS = [accent, accent2, accent3];

  // ===== 1. 散点图 =====
  var raw = window.SCATTER_DATA; // [income, balance_wan, age, cluster]
  var scatter = echarts.init(document.getElementById('chart-scatter'), null, { renderer: 'svg' });
  var series = [0, 1, 2].map(function (c) {
    return {
      name: NAMES[c],
      type: 'scatter',
      data: raw.filter(function (p) { return p[3] === c; })
        .map(function (p) { return [p[0], p[1], p[2]]; }),
      symbolSize: function (v) { return 4 + (v[2] - 18) * 0.35; },
      itemStyle: { color: COLORS[c], opacity: 0.75, borderColor: '#ffffff', borderWidth: 0.3 },
      emphasis: { itemStyle: { opacity: 1 } }
    };
  });
  scatter.setOption({
    animation: false,
    color: COLORS,
    legend: { top: 0, textStyle: { color: muted, fontSize: 12 }, icon: 'circle' },
    grid: { left: 56, right: 24, top: 40, bottom: 44 },
    tooltip: {
      appendToBody: true,
      backgroundColor: '#0b1a33', borderColor: rule, textStyle: { color: ink },
      formatter: function (p) {
        return p.seriesName + '<br/>收入：' + p.value[0].toFixed(1) + ' 万'
          + '<br/>余额：' + p.value[1].toFixed(1) + ' 万'
          + '<br/>年龄：' + p.value[2] + ' 岁';
      }
    },
    xAxis: {
      name: '年收入（万元）', nameTextStyle: { color: muted },
      axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    yAxis: {
      name: '账户余额（万元）', nameTextStyle: { color: muted },
      axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: series
  });
  window.addEventListener('resize', function () { scatter.resize(); });

  // ===== 2. 饼图 =====
  var pie = echarts.init(document.getElementById('chart-pie'), null, { renderer: 'svg' });
  pie.setOption({
    animation: false,
    color: [accent, accent3, accent2],
    tooltip: { appendToBody: true, backgroundColor: '#0b1a33', borderColor: rule, textStyle: { color: ink } },
    legend: { bottom: 0, textStyle: { color: muted, fontSize: 12 } },
    series: [{
      type: 'pie',
      radius: ['42%', '68%'],
      center: ['50%', '46%'],
      label: { color: ink, formatter: '{b}\n{d}%（{c}人）', fontSize: 12 },
      labelLine: { lineStyle: { color: muted } },
      itemStyle: { borderColor: '#050d1f', borderWidth: 2 },
      data: [
        { name: '年轻潜力客群', value: 126 },
        { name: '主力中产客群', value: 124 },
        { name: '富裕稳健客群', value: 50 }
      ]
    }]
  });
  window.addEventListener('resize', function () { pie.resize(); });

  // ===== 3. 雷达图 =====
  var radar = echarts.init(document.getElementById('chart-radar'), null, { renderer: 'svg' });
  radar.setOption({
    animation: false,
    color: COLORS,
    tooltip: { appendToBody: true, backgroundColor: '#0b1a33', borderColor: rule, textStyle: { color: ink } },
    legend: { bottom: 0, textStyle: { color: muted, fontSize: 12 } },
    radar: {
      center: ['50%', '48%'], radius: '62%',
      indicator: [
        { name: '年收入', max: 1 },
        { name: '账户余额', max: 1 },
        { name: '年龄', max: 1 },
        { name: '持有产品数', max: 1 }
      ],
      axisName: { color: ink, fontSize: 12 },
      splitLine: { lineStyle: { color: rule } },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      data: [
        // 真实均值的 min-max 归一化：收入[6.11,18.94,41.58] 余额[1.45,12.45,48.35] 年龄[21.49,37.28,47.10] 产品[1.96,2.18,1.96]
        { name: '年轻潜力客群', value: [0.00, 0.00, 0.00, 0.00], areaStyle: { opacity: 0.15 } },
        { name: '富裕稳健客群', value: [1.00, 1.00, 1.00, 0.00], areaStyle: { opacity: 0.15 } },
        { name: '主力中产客群', value: [0.36, 0.23, 0.62, 1.00], areaStyle: { opacity: 0.15 } }
      ]
    }]
  });
  window.addEventListener('resize', function () { radar.resize(); });
})();
