import getDistance from './calculateDistance'

const maxLng = 115; const minLng = 110; const maxLat = 25; const minLat = 20;
const gridSize = 100;//格网大小
const lngDis = getDistance(maxLng, minLat, minLng, minLat);//经度距离
const latDis = getDistance(minLng, maxLat, minLng, minLat);//纬度距离
const lngRate = (maxLng - minLng) * gridSize / lngDis; //经度比例，每个格网占多少度
const latRate = (maxLat - minLat) * gridSize / latDis; //纬度比例，每个格网占多少度

function getGridIndex(lng, lat) {
    const lngIndex = Math.ceil(getDistance(lng, minLat, minLng, minLat) / gridSize);//计算lng方向上的格网编号，向上取整
    const latIndex = Math.ceil(getDistance(minLng, lat, minLng, minLat) / gridSize);//计算lat方向上的格网编号，向上取整
    let gridCenterLng = minLng + (lngIndex - 0.5) * lngRate; //计算格网中心的经度
    let gridCenterLat = minLat + (latIndex - 0.5) * latRate; //计算格网中心的纬度
    return [gridCenterLng, gridCenterLat];//返回格网中心经纬度坐标
}
export default function getWeightData(allUserNodes) {
    const newData = {};
    const weightData = new Array();
    let gridLngNums = Math.ceil(lngDis / gridSize);
    let gridLatNums = Math.ceil(latDis / gridSize);
    // console.log(gridLngNums, gridLatNums);
    for (let i = 0; i < gridLngNums; i++){
        for(let j = 0; j < gridLatNums; j++){
            console.log('111')
            newData['['+(i + 0.5) * lngRate.toString() + ',' + (j + 0.5) * latRate.toString() + ']'] = 0;
        }
    }
    console.log(newData);
    for (let i = 0; i < allUserNodes.length; i++){
        let lng = allUserNodes[i].COORDINATE[0];
        let lat = allUserNodes[i].COORDINATE[1];
        let [gridCenterLng, gridCenterLat] = getGridIndex(lng, lat);
        newData['[' + gridCenterLng.toString() + ',' + gridCenterLat.toString() + ']'] += 1;
    }
    for (let i = 0; i < Object.keys(newData).length; i++){
        if(Object.values(newData)[i] !== 0){
            weightData.push({COORDINATE: Object.keys(newData)[i], WEIGHT: Object.values(newData)[i]});
        }
    }
    console.log(weightData);
    return weightData;//格网中心点的经纬度及其weight
}