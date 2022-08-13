// 计算球面距离函数
import { getFlatternDistance } from './getFlatternDistance';
import _ from 'lodash';



// 统计 O / D 落入 poi块 的轨迹编号
export const getGridODs = (userData, cellCenter, dis) => {
    const [centerLng, centerLat] = cellCenter;
    console.log(centerLng, centerLat)
    const selectedIDs = userData.reduce((prev, cur) => {
        const isPush = getFlatternDistance(centerLat, centerLng, cur?.origin?.[1], cur?.origin?.[0]) <= dis ||
            getFlatternDistance(centerLat, centerLng, cur?.destination?.[1], cur?.destination?.[0]) <= dis;
        if (isPush) {
            prev.push(cur.id);
        }
        return prev;
    }, [])
    return selectedIDs;
}