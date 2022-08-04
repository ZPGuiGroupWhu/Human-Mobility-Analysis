var EARTH_RADUIS = 6378137.0; //单位M
var PI = Math.PI;

function getRad(d) {
    return d * PI / 180.0;
}

//计算球面两个位置的距离
export default function getDistance(lng1, lat1, lng2, lat2) {
    let radLat1 = getRad(lat1);
    let radLat2 = getRad(lat2);
    let a = getRad(lat1 - lat2);
    let b = getRad(lng1 - lng2);
    let s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) + Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)));
    s = s * EARTH_RADUIS ;// EARTH_RADIUS;
    s = Math.round(s * 10000) / 10000;
    return s;
}