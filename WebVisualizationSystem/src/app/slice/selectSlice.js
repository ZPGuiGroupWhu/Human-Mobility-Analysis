import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';
import { getOceanScoreAll, getUsersTopFive, getUserTrajNumsByDay } from '@/network';

const initialState = {
  curId: -1, // box-chart 实例唯一标识
  OceanReqStatus: '',
  LocationsReqStatus: '',
  CountsReqStatus: '',
  data: null,
  OceanScoreAll: null, // chart 图表数据源 -- 大五人格
  UsersTopFive: null, // 地图数据源 -- top5 位置
  UserTrajNumsByDay: null, // 日历 数据源
  selectedUsers: [], // 筛选得到的用户编号数组(交集)
  selectedByHistogram: [], // 柱状图筛选结果
  selectedByScatter: [], // 散点图筛选结果
  selectedByParallel: [], // 平行坐标轴筛选结果
  selectedByCharts: [], // 图表筛选结果, 三个结果的交集
  selectedByCalendar: [], // 日历筛选结果
  selectedByMapBrush: [], // 地图筛选结果
  selectedByMapClick: [], // 地图点击结果
}

// 根据 url 地址获取数据
const fetchData = createAsyncThunk('select/fetchData', async function (url) {
  const resObj = await axios.get(url);
  return resObj.data;
});

// 获取用户 top 5 位置数据
const fetchUsersTopFive = createAsyncThunk('select/fetchUsersTopFive', async function(){
  const { data } = await getUsersTopFive();
  return data;
})
// 获取各天各用户出行次数
const fetchUserTrajNumsByDay = createAsyncThunk('select/fetchUserTrajNumsByDay', async function(){
  const { data } = await getUserTrajNumsByDay();
  return data;
})
// 获取全部大五人格数据
const fetchOceanScoreAll = createAsyncThunk('select/fetchOceanScoreAll', async function () {
  const { data }= await getOceanScoreAll();
  return data;
})

const selectReducer = createSlice({
  name: 'select',
  initialState,
  reducers: {
    setSelectedUsers: (state, action) => {
      state.selectedUsers = action.payload;
      console.log('selected users', action.payload)
    },
    setSelectedByHistogram: (state, action) => {
      state.selectedByHistogram = action.payload;
    },
    setSelectedByScatter: (state, action) => {
      state.selectedByScatter = action.payload;
    },
    setSelectedByParallel: (state, action) => {
      state.selectedByParallel = action.payload;
    },
    setSelectedByCharts: (state, action) => {
      state.selectedByCharts = action.payload;
    },
    setSelectedByCalendar: (state, action) => {
      state.selectedByCalendar = action.payload;
    },
    setSelectedByMapBrush: (state, action) => {
      state.selectedByMapBrush = action.payload;
    },
    setSelectedByMapClick: (state, action) => {
      state.selectedByMapClick = action.payload;
    },

    setCurId: (state, action) => {
      state.curId = action.payload;
    },
  },
  extraReducers: {
    // fetchData
    [fetchData.pending]: (state) => {
      state.OceanReqStatus = 'loading';
      console.log('ocean loading...')
    },
    [fetchData.fulfilled]: (state, action) => {
      state.OceanScoreAll = action.payload;
      state.OceanReqStatus = 'succeeded';
      console.log('ocean success...')
    },
    [fetchData.rejected]: (state, action) => {
      state.OceanReqStatus = 'failed';
      throw new Error(action.error.message);
    },
    // fetchOceanScoreAll
    [fetchOceanScoreAll.pending]: (state) => {
      state.OceanReqStatus = 'loading';
    },
    [fetchOceanScoreAll.fulfilled]: (state, action) => {
      state.OceanScoreAll = action.payload;
      state.OceanReqStatus = 'succeeded';
    },
    [fetchOceanScoreAll.rejected]: (state, action) => {
      state.OceanReqStatus = 'failed';
      throw new Error(action.error.message);
    },
     // fetchUsersTopFive
     [fetchUsersTopFive.pending]: (state) => {
      console.log('topFive loading...')
      state.LocationsReqStatus = 'loading';
    },
    [fetchUsersTopFive.fulfilled]: (state, action) => {
      state.UsersTopFive = action.payload;
      state.LocationsReqStatus = 'succeeded';
      console.log('topFive success...')
    },
    [fetchUsersTopFive.rejected]: (state, action) => {
      state.LocationsReqStatus = 'failed';
      throw new Error(action.error.message);
    },
     // fetchUserTrajNumsByDay
     [fetchUserTrajNumsByDay.pending]: (state) => {
      state.CountsReqStatus = 'loading';
      console.log('dateCounts loading...')
    },
    [fetchUserTrajNumsByDay.fulfilled]: (state, action) => {
      state.UserTrajNumsByDay = action.payload;
      state.CountsReqStatus = 'succeeded';
      console.log('dateCounts success...')
    },
    [fetchUserTrajNumsByDay.rejected]: (state, action) => {
      state.CountsReqStatus = 'failed';
      throw new Error(action.error.message);
    },
  }
});

export {
  fetchData,
  fetchOceanScoreAll,
  fetchUsersTopFive,
  fetchUserTrajNumsByDay,
};

export const { 
  setSelectedUsers, 
  setSelectedByHistogram,
  setSelectedByScatter,
  setSelectedByParallel,
  setSelectedByCharts, 
  setSelectedByCalendar, 
  setSelectedBySlider,
  setSelectedByMapBrush,
  setSelectedByMapClick, 
  setCurId,
} = selectReducer.actions;

export default selectReducer.reducer;