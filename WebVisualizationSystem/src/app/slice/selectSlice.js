import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';
import { getOceanScoreAll } from '@/network';

const initialState = {
  curId: -1, // box-chart 实例唯一标识
  reqStatus: '',
  data: null, // chart 图表数据源
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

// 获取全部大五人格数据
const fetchOceanScoreAll = createAsyncThunk('select/fetchOceanScoreAll', async function () {
  const data = await getOceanScoreAll();
  console.log(data);
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
      state.reqStatus = 'loading';
    },
    [fetchData.fulfilled]: (state, action) => {
      state.data = action.payload;
      state.reqStatus = 'succeeded';
    },
    [fetchData.rejected]: (state, action) => {
      state.reqStatus = 'failed';
      throw new Error(action.error.message);
    },
    // fetchOceanScoreAll
    [fetchOceanScoreAll.pending]: (state) => {
      state.reqStatus = 'loading';
    },
    [fetchOceanScoreAll.fulfilled]: (state, action) => {
      state.data = action.payload;
      state.reqStatus = 'succeeded';
    },
    [fetchOceanScoreAll.rejected]: (state, action) => {
      state.reqStatus = 'failed';
      throw new Error(action.error.message);
    },
  }
});

export {
  fetchData,
  fetchOceanScoreAll,
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