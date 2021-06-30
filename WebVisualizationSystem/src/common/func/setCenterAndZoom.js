import BMap from 'BMap';

// 获取缩放级别
function getZoom(map, maxLng, minLng, maxLat, minLat) {
  let zoom = ["50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000", "25000", "50000", "100000", "200000", "500000", "1000000", "2000000"];// 级别18到3。
  let pointMax = new BMap.Point(maxLng, maxLat);
  let pointMin = new BMap.Point(minLng, minLat);
  let dist = map.getDistance(pointMax, pointMin).toFixed(1);
  for (let i = 0; i < zoom.length; i++) {
    if (zoom[i] - dist > 0) {
      return 18 - i + 4;
    }
  }
}

/**
 * 居中定位
 * @param {object} map - 地图实例
 * @param {number[]} points - 坐标数组
 */
export function setCenterAndZoom(map, points, maxLength = 1000) {
  if (points.length > 0) {
    // 分段计算，防止栈内存溢出
    const lens = points.length;
    const chunks = Math.ceil(lens / maxLength);
    // 存储已计算部分的最值
    let maxLng = -Infinity,
      minLng = Infinity,
      maxLat = -Infinity,
      minLat = Infinity;
    for (let i = 0; i < chunks; i++) {
      let lngs = [], lats = [];
      let start = i * maxLength, end = (i + 1) * maxLength;
      for (let i = start; i < end; i++) {
        if (i >= lens) break;
        lngs.push(points[i][0]);
        lats.push(points[i][1]);
      }
      maxLng = Math.max(...lngs, maxLng);
      minLng = Math.min(...lngs, minLng);
      maxLat = Math.max(...lats, maxLat);
      minLat = Math.min(...lats, minLat);
    }

    // let res;
    // for (let i = points.length - 1; i >= 0; i--) {
    //   res = points[i];
    //   if (res[0] > maxLng) maxLng = res[0];
    //   if (res[0] < minLng) minLng = res[0];
    //   if (res[1] > maxLat) maxLat = res[1];
    //   if (res[1] < minLat) minLat = res[1];
    // }
    let cenLng = (parseFloat(maxLng) + parseFloat(minLng)) / 2;
    let cenLat = (parseFloat(maxLat) + parseFloat(minLat)) / 2;
    let zoom = getZoom(map, maxLng, minLng, maxLat, minLat);
    map.centerAndZoom(new BMap.Point(cenLng, cenLat), zoom);
  } else {
    //没有坐标，显示全中国
    map.centerAndZoom(new BMap.Point(103.388611, 35.563611), 5);
  }
}