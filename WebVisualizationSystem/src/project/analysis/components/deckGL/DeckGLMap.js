import React, { useState, useEffect, useRef, useMemo, useLayoutEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { StaticMap } from 'react-map-gl';
import { HeatmapLayer, GPUGridLayer, CPUGridLayer } from '@deck.gl/aggregation-layers';
import { ArcLayer, IconLayer } from '@deck.gl/layers';
import { TripsLayer } from '@deck.gl/geo-layers';
import _ from 'lodash';
import { Layout, Switch, Slider, Radio, Button, Input, Tooltip, Select, Checkbox, Row, Col, notification } from 'antd';
import { CopyOutlined, SearchOutlined, FrownOutlined, RedoOutlined } from '@ant-design/icons';
import { withRouter } from 'react-router';
// react-redux
import { useSelector, useDispatch } from 'react-redux';
import { setSelectedTraj } from '@/app/slice/predictSlice';
import { addSelectTrajs, setCurShowTrajId, setPoiSelected, singleSelectPoisForPie, doubleSelectPoisForPie, clearPoisForPie, clearPoiSelected } from '@/app/slice/analysisSlice';
// 函数
import { copyText } from '@/common/func/copyText.js';
import { getOneTraj, getUserTrajRegex } from '@/network';
import { getUniqueByKey } from './libs/getUniqueByKey';
import { getPieData } from './libs/gePieData';
import { getGridODsSingle, getGridODsDouble } from './libs/getGridODs';
// 样式
import './DeckGLMap.scss';
import '@/project/border-style.scss';
// hooks
import { useView } from './hooks/useView';
import { useSelectLogic } from "./hooks/useSelectLogic";
import { useHistory } from 'react-router-dom';
// 配置
import { INITIAL_VIEW_STATE, MAPBOX_ACCESS_TOKEN, POI_COLOR_RANGE } from './configs/poiMap/Config';
// components
import PoiPie from './components/poi-pie/PoiPie';
import { poiMapTooltip, speedTooltip, gridTooltip } from './configs/Tooltips';


const tripInitOpacity = 0.8;
const iconInitOpacity = 256;


function DeckGLMap(props) {

  const dispatch = useDispatch();

  const analysis = useSelector(state => state.analysis);
  const predict = useSelector(state => state.predict);

  const { userId, userData, getTrajCounts, setRoutes, pois } = props;

  const history = useHistory(); // hooks 路由传参数方法
  const inputRef = useRef(); // 文本框ref对象

  // 视角操作逻辑
  const { prevViewState, flyToFocusPoint } = useView(INITIAL_VIEW_STATE);

  // 双选逻辑 (支持 Shift 快捷键)
  const { checkedOptions, checkedValue, setCheckedValue, onCheckboxValueChange } = useSelectLogic();

  const { Sider } = Layout;

  const ICON_MAPPING_O = {
    marker: { x: 0, y: 0, width: 200, height: 200, mask: true }
  };

  const ICON_MAPPING_D = {
    marker: { x: 0, y: 0, width: 200, height: 250, mask: true }
  };

  const [heatMapLayerShow, setHeatMapLayerShow] = useState(false);
  const [gridLayerShow, setGirdLayerShow] = useState(false);
  const [gridLayer3D, setGridLayer3D] = useState(true);
  const [speedLayerShow, setSpeedLayerShow] = useState(false);
  const [speedLayer3D, setSpeedLayer3D] = useState(true);
  const [tripsLayerShow, setTripsLayerShow] = useState(true);
  const [iconLayerOShow, setIconLayerOShow] = useState(false);
  const [iconLayerDShow, setIconLayerDShow] = useState(false);
  const [tripsLayerOneShow, setTripsLayerOneShow] = useState(true);
  const [arcLayerOneShow, setArcLayerOneShow] = useState(true);
  const [iconLayerOneOShow, setIconLayerOneOShow] = useState(true);
  const [iconLayerOneDShow, setIconLayerOneDShow] = useState(true);
  const [poiMapShow, setPoiMapShow] = useState(false);
  const [gridWidth, setGridWidth] = useState(100);
  const [tripsOpacity, setTripsOpacity] = useState(tripInitOpacity); // 轨迹初始透明度
  const [iconOpacity, setIconOpacity] = useState(iconInitOpacity); // icon图标图层初始化透明度
  const [trajIdForSearch, setTrajIdForSearch] = useState(''); // 定向查找输入的轨迹编号字符串
  const [trajIdForSelect, setTrajIdForSelect] = useState([]); // 轨迹编号字符串的模糊检索结果

  const record = useRef({
    iconDisabled: false, // icon图层开关的disabled属性
    iconChecked: false, // icon图层开关属性
  })


  const returnSelectTrajs = (selectTrajIds) => {
    let selectTrajs = [];
    _.forEach(userData, (item) => {
      if (selectTrajIds.includes(item.id)) {
        selectTrajs.push(dataFormat(item));
      }
    })
    return selectTrajs; // 返回选择的轨迹信息
  }

  const dataFormat = (traj) => {
    let path = []; // 组织为经纬度数组
    let importance = []; // 存储对应轨迹每个位置的重要程度
    for (let j = 0; j < traj.lngs.length; j++) {
      path.push([traj.lngs[j], traj.lats[j]]);
      // 计算重要程度，判断speed是否为0
      importance.push(
        traj.spd[j] === 0 ?
          traj.azimuth[j] * traj.dis[j] / 0.00001 :
          traj.azimuth[j] * traj.dis[j] / traj.spd[j]);
    }
    // 组织数据, 包括id、date(用于后续选择轨迹时在calendar上标记)、data(轨迹）、spd（轨迹点速度）、azimuth（轨迹点转向角）、importance（轨迹点重要程度）
    let res = {
      id: traj.id,
      date: traj.date,
      data: path,
      spd: traj.spd,
      azimuth: traj.azimuth,
      importance: importance,
      // 新添加了细粒度时间特征
      weekday: traj.weekday + 1,
      hour: traj.hour,
    };
    return res;
  }

  // 根据当前轨迹 id 返回信息
  const handleCurTrajId = (selectTrajs, curShowTrajId) => {
    if (selectTrajs.length && curShowTrajId !== -1) {
      console.log(selectTrajs)
      console.log(curShowTrajId)
      const trajs = selectTrajs.find(item => item.id === curShowTrajId);
      if (!trajs) {
        return {}
      } else {
        const O = trajs.data[0], D = trajs.data.slice(-1)[0];
        const trajObj = {
          O: [{ COORDINATES: O }],
          D: [{ COORDINATES: D }],
          OD: [{ O: O, D: D }],
          Path: [{ path: trajs.data }],
        };
        return trajObj;
      }
    } else {
      return {}
    }
  }

  // 筛选 cell 单元格内占比最大的 type 类别 对应的 序号，返回颜色值
  const getColorValueByPoiGridLayer = (points) => {
    const map = _.countBy(points, (item) => item.typeId);
    const idx = Object.values(map).findIndex(
      (item, idx, arr) => item === Math.max(...arr)
    );
    const value = poiTypes.findIndex((type) => {
      return +Object.keys(map)[idx] === type;
    });
    return value;
  };

  // 基于 cell 单元格内聚合的 POI 点数返回对应的 bar height
  const getElevationValueByPoiGridLayer = (points) => points.length;

  // 根据图层显示与否，决定 tooltips 的格式
  const getTooltips = () => {
    if (gridLayerShow) {
      return gridTooltip.getTooltip;
    }
    else if (speedLayerShow) {
      return speedTooltip.getTooltip;
    }
    else if (poiMapShow) {
      return poiMapTooltip.getTooltip;
    }

  };

  // 是否开启双选
  const isDoubleSelected = useMemo(
    () => {
      return checkedValue.includes("double-select")
    },
    [checkedValue]
  );

  // 处理轨迹点、统计轨迹数目
  const { trajCounts, trajNodes } = useMemo(() => {
    const Nodes = [];//统计所有节点的坐标
    const Counts = {};//统计每天的轨迹数目
    for (let i = 0; i < userData.length; i++) {
      if (Counts[userData[i].date] === undefined) {
        Counts[userData[i].date] = { 'count': 1 }//若当天没有数据，则表明是第一条轨迹被录入
      }
      else {
        Counts[userData[i].date] = { 'count': Counts[userData[i].date].count + 1 }//否则是其他轨迹被录入，数目加一
      }
      for (let j = 0; j < userData[i].lngs.length; j++) {
        Nodes.push({ COORDINATES: [userData[i].lngs[j], userData[i].lats[j]], WEIGHT: 1, SPD: userData[i].spd[j] });//将所有轨迹点放入同一个数组内，权重均设置为1
      }
    }
    getTrajCounts(Counts); //将每天的轨迹数目统计结果反馈给父组件
    const obj = {
      trajCounts: Counts,
      trajNodes: Nodes,
    }
    return obj
  }, [userData]);


  // 将筛选的轨迹编号处理为 tripslayer 需要的轨迹数据
  const selectData = useMemo(() => {
    return returnSelectTrajs(analysis.finalSelected);
  }, [analysis.finalSelected])


  // 展示的当前选择的轨迹
  const curShowTraj = useMemo(() => {
    // 如果清空购物车，恢复原始 opacity
    if (!analysis.selectTrajs.length) {
      setTripsOpacity(tripInitOpacity);
      setIconOpacity(iconInitOpacity);
    }
    return handleCurTrajId(analysis.selectTrajs, analysis.curShowTrajId);
  }, [analysis.curShowTrajId, analysis.selectTrajs])

  // poi 类别
  const poiTypes = useMemo(
    () =>
      getUniqueByKey(pois, function (poi) {
        return poi?.typeId;
      }),
    [pois]
  );

  // poi块 弧线图层
  const poiArcData = useMemo(() => {
    let res = [];
    if (analysis.poisForPie.length % 2 === 0) {
      for (let i = 0; i < analysis.poisForPie.length; i += 2) {
        const from = {
          name: "起点",
          location: analysis.poisForPie?.[i]?.[0]?.cellCenter,
        };
        const to = {
          name: "终点",
          location: analysis.poisForPie?.[i + 1]?.[0]?.cellCenter,
        };
        res.push({
          from,
          to,
        });
      }
    }
    return res;
  }, [analysis.poisForPie]);

  // PoiMap 三维柱状图
  const PoiMapLayer = new CPUGridLayer({
    id: "poi-map",
    visible: poiMapShow,
    colorDomain: [0, 11],
    colorRange: POI_COLOR_RANGE,
    data: pois,
    pickable: true,
    extruded: true,
    cellSize: 400,
    elevationScale: 4,
    getPosition: (d) => d.location,
    getColorValue: getColorValueByPoiGridLayer,
    getElevationValue: getElevationValueByPoiGridLayer,
    onClick: (info) => {
      // console.log(info)
      const data = getPieData(info);
      if (!isDoubleSelected) {
        dispatch(singleSelectPoisForPie(Object.values(data)));
        flyToFocusPoint(info?.object?.position);
      } else {
        dispatch(doubleSelectPoisForPie(Object.values(data)));
      }
    },
  })

  // poi连线图层
  const poiArcLayer = new ArcLayer({
    id: "poi-arc",
    data: poiArcData,
    pickable: true,
    getWidth: 3,
    greatCircle: true,
    getSourcePosition: (d) => d.from.location,
    getTargetPosition: (d) => d.to.location,
    getSourceColor: [255, 241, 240],
    getTargetColor: [255, 77, 79],
  });

  //轨迹热力图图层
  const heatMapLayerGrid = new HeatmapLayer({
    id: 'heatmapLayer',
    visible: gridLayerShow && heatMapLayerShow,
    data: trajNodes,
    getPosition: d => d.COORDINATES,
    getWeight: d => d.WEIGHT,
    aggregation: 'SUM'
  });

  // 速度热力图图层
  const heatMapLayerSPD = new HeatmapLayer({
    id: 'heatmapLayerSPD',
    visible: speedLayerShow && heatMapLayerShow,
    radiusPixels: 3,  // 速度图层带宽
    data: trajNodes,
    getPosition: d => d.COORDINATES,
    getWeight: d => d.SPD,
    aggregation: 'MEAN'
  });

  //构建轨迹格网图层
  const gridLayer = new GPUGridLayer({
    id: 'gpu-grid-layer',
    visible: gridLayerShow && !heatMapLayerShow,
    data: trajNodes,
    pickable: true,
    extruded: gridLayer3D, // 是否显示为3D效果
    cellSize: gridWidth, // 格网宽度，默认为100m
    elevationScale: 4,
    colorRange: [[171, 217, 233], [224, 243, 248], [255, 255, 191], [254, 224, 144], [253, 174, 97], [244, 109, 67]],
    getPosition: d => d.COORDINATES,
  });

  //构建速度图层
  const speedLayer = new GPUGridLayer({
    id: 'gpu-grid-layer-speed',
    visible: speedLayerShow && !heatMapLayerShow,
    data: trajNodes,
    pickable: true,
    extruded: speedLayer3D,//是否显示为3D效果
    cellSize: gridWidth,//格网宽度，默认为100m
    elevationScale: 4,
    elevationAggregation: 'MEAN',//选用速度均值作为权重
    colorAggregation: 'MEAN',
    colorRange: [[171, 217, 233], [224, 243, 248], [255, 255, 191], [254, 224, 144], [253, 174, 97], [244, 109, 67]],
    getPosition: d => d.COORDINATES,
    getElevationWeight: d => d.SPD,
    getColorWeight: d => d.SPD,
  });

  const iconLayerOneO = new IconLayer({
    id: 'icon-layer-one-O',
    visible: iconLayerOneOShow,
    data: curShowTraj.O,
    pickable: true,
    // iconAtlas和iconMapping必须，iconAtlas=>base64形式
    // getIcon: return a string
    iconAtlas: 'data:image/png;base64,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',
    iconMapping: { marker: { x: 0, y: 0, width: 200, height: 200, mask: true } },
    getIcon: d => 'marker',
    getPosition: d => d.COORDINATES,
    sizeScale: 20,
    getSize: d => 1,
    getColor: d => [175, 238, 238],
  });

  const iconLayerOneD = new IconLayer({
    id: 'icon-layer-one-D',
    visible: iconLayerOneDShow,
    data: curShowTraj.D,
    pickable: true,
    // iconAtlas和iconMapping必须，iconAtlas=>base64形式
    // getIcon: return a string
    iconAtlas: 'data:image/png;base64,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',
    iconMapping: { marker: { x: 0, y: 0, width: 200, height: 250, mask: true } },
    getIcon: d => 'marker',
    getPosition: d => d.COORDINATES,
    sizeScale: 20,
    getSize: d => 1,
    getColor: d => [255, 69, 0],
  });

  const arcLayerOne = new ArcLayer({
    id: 'arc-layer-one',
    visible: arcLayerOneShow,
    data: curShowTraj.OD,
    pickable: true,
    getWidth: 1,
    getSourcePosition: d => d.O,
    getTargetPosition: d => d.D,
    getSourceColor: [175, 255, 255],
    getTargetColor: [0, 128, 128],
  });

  const tripsLayerOne = new TripsLayer({
    id: 'trips-layer-one',
    visible: tripsLayerOneShow,
    data: curShowTraj.Path,
    getPath: d => d.path,
    // deduct start timestamp from each data point to avoid overflow
    // getTimestamps: d => d.waypoints.map(p => p.timestamp - 1554772579000),
    getColor: [256, 0, 0],
    opacity: 1,
    widthMinPixels: 3,
    rounded: true,
    trailLength: 200,
    currentTime: 100,
  });

  // 绘制轨迹图层
  const tripsLayer = new TripsLayer({
    id: 'trips-layer',
    visible: tripsLayerShow,
    data: selectData,
    getPath: d => d.data,
    // deduct start timestamp from each data point to avoid overflow
    // getTimestamps: d => d.waypoints.map(p => p.timestamp - 1554772579000),
    getColor: [244, 164, 96],
    // opacity: 512/ this.state.dataLength,
    opacity: tripsOpacity,
    widthMinPixels: 3,
    rounded: true,
    fadeTrail: true,
    trailLength: 200,
    currentTime: 100,
    pickable: true,
    autoHighlight: true,
    shadowEnabled: false,
    highlightColor: [256, 0, 0, 256],
    onClick: info => clickEvents(info),
  })


  //O点icon图层
  const iconLayerO = new IconLayer({
    id: 'icon-layer-O',
    visible: iconLayerOShow,
    data: selectData,
    pickable: true,
    opacity: iconOpacity,
    // iconAtlas和iconMapping必须，iconAtlas=>base64形式
    // getIcon: return a string
    iconAtlas: 'data:image/png;base64,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',
    iconMapping: ICON_MAPPING_O,
    getIcon: d => 'marker',
    sizeScale: 20,
    onClick: info => clickEvents(info),
    getPosition: d => d.data[0],  // 轨迹第一个点坐标
    getSize: d => 1,
    getColor: d => [175, 238, 238],
  })

  //D点icon图层
  const iconLayerD = new IconLayer({
    id: 'icon-layer-D',
    visible: iconLayerDShow,
    data: selectData,
    pickable: true,
    opacity: iconOpacity,
    // iconAtlas和iconMapping必须，iconAtlas=>base64形式
    // getIcon: return a string
    iconAtlas: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAAAXNSR0IArs4c6QAAC7NJREFUeF7tnHuobGUdhh/tomSpmJpGWSgoViplEV6yY3Yx7WaFYpmWJXYxu5cW9VdRBpmVFBn5RymZQmmek1qYhldC7SKWWWml5KWii6WdEIpffUPTdm/PnJk1e95Z8yxYzDmcWd96v+ed56yZtdbMJqy8PAOodff2uDdwJ3A9cCPwU+CCh9i+j/+0BfAU4KnAkx5igrcDtf62Pd7XRxiLMKdNVpjkp4ETgIdvAML3gOOBX/YQ1qOBA4HnNCFKjCePOc8/NFGuA74NfB/405hjudkqElgqyDOBi4HHbmSGE4HPbeQ2aU/fHNgHWAMcBOw35YDfBS4DrgCunPK+HH5MAksF+deY49Rm9Rbshgm2X+1NHwbs29YDmhSbrXaItr+fA99qq7LMqITldjssyEnAxyfIdld7C7J+gjGmvemz25GhxKi3TttPe4djjH/tkCw3jbG9m3RIYCBI/Q9a74snXT4PvG3SQTrcvo4Sz2/rC4C9Ohx7NYZaC3wZOH81duY+HkxgIMgXgDd3BGgX4NaOxhpnmG2AkmEgxrgfrMfZ97S2ubSJ8rVp7cBxlycwEOSq9l68C06vmMHp352HjhIlxtZdTCRwjKubKGcGZutlpIEg9wJ1WrOL5cPAR7sYaANj7Ai8GnhxW1dhlzG7uAaoU/HnxSTqZ5A1JUhdCKyLfl0t5wJHdDXYMuPUW8GXLaAUyyEtQUqUEsalWwL1rupDJUid96/z8V0tl7cLbF2NV+Mc2j7819HC5cEESpJTgTuEMzGBTdvlijqhc2C6IGcAx0085cUY4DbgI8BZizHdqcyyruXVafbBHSTxgtSRrY5wLqMT+GIT5Z7RN/GZwJuALy0hoSA9fWnUBcY6mnyjp/Prclp1WaBOKr1lmUEVpEvSgWOdDpziZ5MVmzkKOLndob3ckxQk8EXddaTfNEnqYrDLfwns0cQ4cgNAFGSBXjEXAZ8E6izjIi8faHJsNQIEBRkBUt+eUqeD623Xon2IP7iJUfcdjrooyKikeva8XzRJ6kbIvi/1HZ9j21mqjZ2rgmwssZ49f127bnJOz+ZV06l78t4AvGaCuSnIBPD6tOkPmyhnA3fP+cQOA44BXt7BPBSkA4h9GqLkKEnqanxJMy9LfV36dcDRwP4dhlaQDmH2bah621WiXAI8EDq53YBXNTHqz10vCtI10R6OVz/1VLf81Onh+pmnWZ/9qq9Nv7TdEFtfnZ7moiDTpNvTsetXWOrDfcly8yrMcdf2gyD1KzMlxk6rsM/BLhRkFWH3cVf121719eqV1lHnXBft6qemtgXqK9t1V+1gfcyog0zheQoyBagO+T8C9wPDa/3C5ODvJcNAikeEQlOQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEEyejBFKAEFCS3GWBkEFCSjB1OEElCQ0GKMlUFAQTJ6MEUoAQUJLcZYGQQUJKMHU4QSUJDQYoyVQUBBMnowRSgBBQktxlgZBBQkowdThBJQkNBijJVBQEHG6OEe4Dbg/hW23RJ4IrDdGGO7SRYBBXmIPn4GXA3c2IS4tT3+fcQONwOe0GSpx1r3Ap4F7DLiGD5ttgQUZIj/D4CrgGvaescUu9kR2HvJ+vgp7s+hxyOw8ILcBKxt65XjMexsq8OAVwL1uEVnozrQJAQWUpC/AWcB3wS+Mwm9KW27U5OkRHnulPbhsKMRWChBfgV8tclRf56HZb8myzHAtvMQuGcZF0KQ+rD92SZGHT3mcdkNeB/wxnkMP8eZey3IfcCpwKeAP89xScPRX9JEOaAn80mfRm8Fqc8YJcaP0hsYM9+7myg7jLm9m41GoHeCXNGOGuePNv+5ftbOwPuB4+d6FtnheyPI7U2M07J5TyXdwe1o8rypjL7Yg/ZCkAuBdwHzcmZqWi+5E5sodcXepRsCcy/IKcBJ3bDoxSh1DaXOdp3Qi9nMfhJzK8gfgfqg+pXZM4xMUG+33gMcEpluPkLVKfUzNwHWAJd1mPly4MCOxqtclW94qQ/iJcd1He2jz8NUySXK7n2eZMdzuwV4K3BpjTtvgpwHvB6oaxwuoxHYGnhvE2Xz0TZZ2Gd9HTi53bX9HwjzJEjJcfjCVjf5xJ8OvBM4evKhejfCeuCD7Uzo/01uXgRRju5ek/W55B3AC7sbcq5Huhj4GLDs3dzzIMjvPXJM5QV4bBNlz6mMnj/oDcBnNnSiJ12Qc5Vjqq+0R7W3XXU1vk4RL8JSX4QrMeoG1n9uaMLpgmwov//eDYH6Hv2Rbe3rd1Dqm6LntLV+V2CkRUFGwrRQT3rRkCyPnPOZPzAkxbpx5qIg41BbjG12baIc2n5oYp5mXXdx1w2rddr25kmCK8gk9BZn2/2BEqXWPUKnfVeT4gKgzkx1sihIJxgXapCDhmSpo8yslzpSlBT12PkX4xRk1vXO9/7raLJ0nebZsJ8APwbqcfDnu6eJUEGmSXcxx35c+4G8ur5St97X37cfelzpFyfranadXVq61g/21WeKWv+x2kgVZLWJu79Nh4Sp3/8qIepi8F8S0ShIYitmiiGgIDFVGCSRgIIktmKmGAIKElOFQRIJKEhiK2aKIaAgMVUYJJGAgiS2YqYYAgoSU4VBEgkoSGIrZoohoCAxVRgkkYCCJLZiphgCChJThUESCShIYitmiiGgIDFVGCSRgIIktmKmGAIKElOFQRIJKEhiK2aKIaAgMVUYJJGAgiS2YqYYAgoSU4VBEgkoSGIrZoohoCAxVRgkkYCCJLZiphgCChJThUESCShIYitmiiGgIDFVGCSRgIIktmKmGAIKElOFQRIJKEhiK2aKIaAgMVUYJJGAgiS2YqYYAgoSU4VBEgkoSGIrZoohoCAxVRgkkYCCJLZiphgCChJThUESCShIYitmiiGgIDFVGCSRgIIktmKmGAIKElOFQRIJKEhiK2aKIVCC7ADc2WGiW4DdOhzPoSQwMwIlSC33ANt1lOJeYMuOxnIYCcyUwECQy4A1HSX5K7BVR2M5jARmSmAgyBnAcR0luQl4WkdjOYwEZkpgIEjJUZJ0sZwNHNXFQI4hgVkTGAhSOa4A9u8g0CHARR2M4xASmDmBYUGOAM6ZMNE1wL4TjuHmEoghMCxIhVoH1BFg3OVw4LxxN3Y7CaQRWCrItsBpwGvHCPp24PQxtnMTCcQSWCrIIGh9aP8EsM0IyS9sH/DXjvBcnyKBuSKwkiA1ib2AfdrjnkCt64HfAb8GrgeuBS6ZqxkbVgIbQeDf6bobNkXKHbEAAAAASUVORK5CYII=',
    iconMapping: ICON_MAPPING_D,
    getIcon: d => 'marker',
    sizeScale: 20,
    onClick: info => clickEvents(info),
    getPosition: d => d.data[d.data.length - 1],  // 轨迹最后一个点坐标
    getSize: d => 1,
    getColor: d => [255, 69, 0],
  })

  //对应新json格式：轨迹点击事件
  const clickEvents = (info) => {
    const id = info.object ? info.object.id : null;
    // 绘制OD弧线
    if (id === null) {
      console.log('no trajectory!')
    } else {
      /**
        * redux-trajs存储
        * info.object = [id:XX, date:XX, data:[[lat1,lng1],[lat2,lng2],....], spd:[spd1,spd2,...], azimuth:[azi1,azi2,...], importance:[imp1,imp2,...]]
        */
      // 1.传递点击选择的轨迹数据
      dispatch(addSelectTrajs(info.object));
      // 2.存储轨迹
      dispatch(setSelectedTraj(info.object));
      // 3.更新当前展示轨迹的 id
      dispatch(setCurShowTrajId(id));

      // 设置透明度
      setTripsOpacity(0.05);
      setIconOpacity(0);

      // 激活“目的地预测”跳转导航
      setRoutes(prev => {
        const newRoutes = _.cloneDeep(prev);
        newRoutes[2].status = true;
        return newRoutes;
      })
    }
  }

  // 切换图层
  const changeGridOrSpeed = (event) => {
    if (event.target.value === "Grid") {
      setGirdLayerShow(true);
      setSpeedLayerShow(false);

    } else if (event.target.value === "Speed") {
      setGirdLayerShow(false);
      setSpeedLayerShow(true);

    } else if (event.target.value === "None") {
      setGirdLayerShow(false);
      setSpeedLayerShow(false);
    }
  }

  const change3D = (event) => { // 切换图层三维显示
    if (event.target.value == "2D") {
      setGridLayer3D(false);
      setSpeedLayer3D(false);
      setHeatMapLayerShow(false);
    }
    else if (event.target.value == "3D") {
      setGridLayer3D(true);
      setSpeedLayer3D(true);
      setHeatMapLayerShow(false);
    }
    else if (event.target.value == "Heat") {
      setHeatMapLayerShow(true);
    }
  };

  const changeGridWidth = (value) => { // 与滑动条联动，切换格网的网格宽度
    setGridWidth(value);
  };

  const changeTripsLayerShow = (event) => { // 与开关联动，切换轨迹图层和icon图层的显示与否
    if (event) { // 点击打开
      record.current.iconDisabled = false; // 取消 icon switch 按钮失效
      // setTripsOpacity(tripInitOpacity);
      // 打开 trips 图层 和 单条轨迹展示图层
      setTripsLayerShow(true);
      setIconLayerOneOShow(true);
      setIconLayerOneDShow(true);
      setArcLayerOneShow(true);
      setTripsLayerOneShow(true);
    } else { // 点击关闭
      // 关掉所有 trips、icon 和 arc 图层，并让 icon 的 switch 按钮失效
      record.current.iconChecked = false;
      record.current.iconDisabled = true;
      setTripsLayerShow(false);
      setIconLayerOShow(false);
      setIconLayerDShow(false);
      setIconLayerOneOShow(false);
      setIconLayerOneDShow(false);
      setArcLayerOneShow(false);
      setTripsLayerOneShow(false);
    }
  }

  //显示和关闭OD点icon图层
  const changeIconLayerShow = (event) => {
    if (event) { // 点击打开
      record.current.iconChecked = true;
      setIconLayerOShow(true);
      setIconLayerDShow(true);
      // setIconOpacity(iconInitOpacity);
    } else { // 点击关闭
      record.current.iconChecked = false;
      setIconLayerOShow(false);
      setIconLayerDShow(false);
    }
  }

  const changePoiMapLayerShow = (event) => {
    if (event) { // 点击打开
      setPoiMapShow(true);
    } else {
      setPoiMapShow(false);
      // 清除数据，并关闭双选
      dispatch(clearPoisForPie());
      setCheckedValue([]);
    }
  }


  const sliderToolTipFormatter = (value) => {
    return `格网宽度：${value}m`
  }


  // 存储输入文本框内的轨迹编号字符串
  const handleChange = _.debounce((value) => { setTrajIdForSearch(value) }, 500);
  const handleSearch = _.debounce(async (value) => {
    let res = await getUserTrajRegex(userId, value);
    setTrajIdForSelect(res);
  }, 500);
  const handleSelect = (value) => {
    new Promise((resolve) => { // 利用 promise 实现 hook setstate 的回调，也可以用 useEffect 实现
      setTrajIdForSearch(value)
    }).then(res => {
      handleSearchTraj()
    });
  }

  // 查找指定的轨迹编号，并保存数据
  const handleSearchTraj = async () => {
    try {
      let data = await getOneTraj(trajIdForSearch);
      data = dataFormat(data);
      // 1.传递点击选择的轨迹数据
      dispatch(addSelectTrajs(data));
      // 2.存储轨迹
      dispatch(setSelectedTraj(data));
      // 3.更新当前展示轨迹的 id
      dispatch(setCurShowTrajId(data.id));

      // 设置透明度
      setTripsOpacity(0.05);
      setIconOpacity(0);
    } catch (err) {
      console.log(err);
    }
  }

  // poi 筛选结果为空 的通知提醒框
  const openNotification = () => {
    notification.info({
      message: '该 OD 格内没有符合要求的轨迹',
      placement: 'topRight',
      top: '5px',
      duration: 1.5,
      style: {
        border: '0.5px solid grey',
        borderRadius: '8px',
        height: '55px',
        fontSize: '10px',
        width: '350px',
      },
      icon: <FrownOutlined style={{ color: '#f7797d' }} />,
    });
  };

  // poi 筛选
  const poiSelect = () => {
    console.log(analysis.poisForPie);
    let result = [];
    if (!isDoubleSelected) {
      result = getGridODsSingle(userData, analysis.poisForPie[0]?.[0]?.cellCenter, 400)
      dispatch(setPoiSelected(result));
    } else {
      result = getGridODsSingle(userData, analysis.poisForPie[0]?.[0]?.cellCenter, 400)
      dispatch(setPoiSelected(result));
    }
    if (!result.length) { // 如果筛选结果为空，则弹出提示框
      openNotification();
    }
  }

  // 双选功能【触发/取消】均重置数据
  useLayoutEffect(() => {
    dispatch(clearPoisForPie());
  }, [isDoubleSelected]);

  // 每次筛选数据都会更新图层颜色透明度
  useEffect(() => {
    setTripsOpacity(tripInitOpacity);
    setIconOpacity(iconInitOpacity);
  }, [analysis.finalSelected])

  // 返回 poiPie sider 右边定位
  useLayoutEffect(() => {
    const right = analysis.selectTrajs.length ? '170px' : '0px';
    document.querySelector('.sider').style.right = right;
  }, [analysis.selectTrajs])

  const selectOptions = trajIdForSelect.sort((a, b) => (a.split('_')[1] - b.split('_')[1]))
    .map(id => <Select.Option key={id}>{id}</Select.Option>); // Select 列表候选项

  // 图层
  const layers = [
    gridLayer,
    heatMapLayerGrid,
    heatMapLayerSPD,
    speedLayer,
    tripsLayer,
    iconLayerO,
    iconLayerD,
    arcLayerOne,
    tripsLayerOne,
    iconLayerOneO,
    iconLayerOneD,
    PoiMapLayer,
    poiArcLayer,
  ]

  return (
    <>
      {/* 主地图 */}
      <DeckGL
        initialViewState={prevViewState}
        controller={true}
        layers={layers}
        getTooltip={getTooltips()}
      >
        {<StaticMap mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN} mapStyle={'mapbox://styles/2017302590157/cksbi52rm50pk17npkgfxiwni'} />}
      </DeckGL>

      {/* 功能栏 */}
      <div style={{ display: 'inline-block' }}>
        <section className='analysis-function-bar moudle' style={{ width: '175px' }}>
          <div className='moudle-white'>
            <Radio.Group
              size='small'
              buttonStyle="solid"
              onChange={changeGridOrSpeed}
              defaultValue="None"
              style={{ marginBottom: '5px' }}
            >
              <Radio.Button value="Grid" >点密度</Radio.Button>
              <Radio.Button value="Speed">速度</Radio.Button>
              <Radio.Button value="None" >关闭</Radio.Button>
            </Radio.Group>
            <Radio.Group
              size='small'
              buttonStyle="solid"
              onChange={change3D}
              defaultValue="3D"
              style={{ marginBottom: '5px' }}
            >
              <Radio.Button value="2D">二维</Radio.Button>
              <Radio.Button value="3D">三维</Radio.Button>
              <Radio.Button value="Heat">热力图</Radio.Button>
            </Radio.Group>
            <Slider
              tipFormatter={sliderToolTipFormatter}
              style={{ width: '93%' }}
              max={500} min={100} step={50}
              defaultValue={300}
              onChange={(value) => changeGridWidth(value)}
            />
          </div>
          <div className={`moudle-white`}>
            <div className='text-button'>
              <span>轨迹图层</span>
              <Switch defaultChecked={true} onChange={changeTripsLayerShow} />
            </div>
            <div className='text-button'>
              <span>OD图层</span>
              <Switch onChange={changeIconLayerShow} disabled={record.current.iconDisabled} checked={record.current.iconChecked} />
            </div>
          </div>
          <div className={`moudle-white`}>
            <div className='text-button'>
              <span>POI图层</span>
              <Switch defaultChecked={false} onChange={changePoiMapLayerShow} />
            </div>
            <div className='poi-function-bar'>
              <Checkbox.Group
                options={checkedOptions}
                value={checkedValue}
                onChange={onCheckboxValueChange}
              />
              <Button
              size='small'
              // ghost
              shape='round'
              icon={<RedoOutlined />}
                onClick={() => {
                  dispatch(clearPoiSelected())
                }}
              > 清除 </Button>

            </div>
          </div>
          {
            Object.keys(predict.selectedTraj).length ?
              <Button type="primary" block onClick={() => {
                history.push('/select/predict');
              }}>目的地预测</Button> : null
          }
        </section>
        <Input.Group compact>
          <Select
            showSearch
            value={trajIdForSearch}
            defaultActiveFirstOption={false}
            showArrow={false}
            filterOption={false}
            onSearch={_.debounce(handleSearch, 500)}
            onChange={_.debounce(handleChange, 500)}
            onSelect={handleSelect}
            notFoundContent={null}
            listHeight={150}
            style={{ width: '120px' }}
          >
            {selectOptions}
          </Select>
          <Tooltip title="拷贝编号">
            <Button icon={<CopyOutlined />} onClick={(e) => { copyText(inputRef.current) }} />
          </Tooltip>
          <Tooltip title="查询并选择">
            <Button
              icon={<SearchOutlined />}
              onClick={(e) => { handleSearchTraj() }}
            />
          </Tooltip>
        </Input.Group>
      </div>
      <Layout>
        <Sider className={'sider'} width={400} theme={"light"}>
          <Row>
            {analysis.poisForPie.map((data, idx) => (
              <Col span={24} key={idx}>
                <PoiPie data={data} />
              </Col>
            ))}
          </Row>
          {analysis.poisForPie.length ?
            <Button
              onClick={() => {
                poiSelect();
              }}
            >筛选轨迹</Button> : null}
        </Sider>
      </Layout>
    </>
  )
}


export default withRouter(DeckGLMap);