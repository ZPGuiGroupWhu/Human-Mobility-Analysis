import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';
import { getOceanScoreAll } from '@/network';

const initialState = {
  curId: -1, // box-chart 实例唯一标识
  reqStatus: '',
  data: null, // chart 图表数据源
  selectedUsers: [], // 筛选得到的用户编号数组(交集)
  selectedByCharts: [], // 图表筛选结果
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
  setSelectedByCharts, 
  setSelectedByCalendar, 
  setSelectedByMapBrush,
  setSelectedByMapClick, 
  setCurId 
} = selectReducer.actions;

export default selectReducer.reducer;