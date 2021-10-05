import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

const initialState = {
  curId: -1, // box-chart 实例唯一标识
  reqStatus: '',
  data: null, // chart 图表数据源
  selectedUsers: [], // 筛选得到的用户编号数组(交集)
  selectedByCharts: [], // 图表筛选结果
  selectedByCalendar: [], // 日历筛选结果
}

const fetchData = createAsyncThunk('select/fetchData', async function (url) {
  const resObj = await axios.get(url);
  return resObj.data;
});

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
    setCurId: (state, action) => {
      state.curId = action.payload;
    },
  },
  extraReducers: {
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
    }
  }
});

export {
  fetchData,
};

export const { setSelectedUsers, setSelectedByCharts, setSelectedByCalendar, setCurId } = selectReducer.actions;

export default selectReducer.reducer;