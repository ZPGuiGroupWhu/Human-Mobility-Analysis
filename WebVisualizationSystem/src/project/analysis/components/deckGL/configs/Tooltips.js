import { getPieData } from "../libs/gePieData";
import { TOOLTIP_STYLE } from "./poiMap/Config";

// gridlayer 图层 Tooltips 格式
export const gridTooltip = {
    getTooltip: (info) => {
        if (!info.picked) return null;
        const lng = (info?.object?.position?.[0]).toFixed(4);
        const lat = (info?.object?.position?.[1]).toFixed(4);
        const count = info?.object?.count;
        return {
            style: TOOLTIP_STYLE,
            html:  `<div><div>经度：${lng}</div><div>纬度：${lat}</div><div>Count：${count}</div></div>`
        }
    }      
}

// speedlayer 图层 Tooltips 格式
export const speedTooltip = {
    getTooltip: (info) => {
        if (!info.picked) return null;
        const lng = (info?.coordinate?.[0]).toFixed(4);
        const lat = (info?.coordinate?.[1]).toFixed(4);
        const speed = (info?.object?.elevationValue).toFixed(4);
        return {
            style: TOOLTIP_STYLE,
            html:  `<div><div>经度：${lng}</div><div>纬度：${lat}</div><div>Speed：${speed}</div></div>`
        }
    } 
}

// poiMap 图层 Tooltips 格式
export const poiMapTooltip = {
    getTooltip: (info) => {
        if (!info.picked) return null;
        const data = getPieData(info);
        return {
          style: TOOLTIP_STYLE,
          html: `<div><div>POI数量：${
            info?.object?.points?.length
          }</div>${Object.entries(data).reduce((prev, cur) => {
            return prev + `<div>${cur[1].type}：${cur[1].value}</div>`;
          }, "")}</div>`,
        };
    }
}