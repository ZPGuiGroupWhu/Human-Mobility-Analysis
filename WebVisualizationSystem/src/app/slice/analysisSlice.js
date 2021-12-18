import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  hourCount: [],
  monthCount: [],
  weekdayCount: [],
  calendarData: [], // 存储 某几个月下的 week-hour 数据
  selectTrajs: [],  // 添加到“轨迹列表”中的轨迹数据集合
  curShowTrajId: -1, // 当前展示的轨迹id
}

const analysisReducer = createSlice({
  name: 'analysis',
  initialState,
  reducers: {
    setAll: (state, action) => {
      ['hourCount', 'monthCount', 'weekdayCount'].forEach((item, idx) => {
        state[item] = action.payload[item]
      })
    },
    setCalendarData: (state, action) => {
      state.calendarData = action.payload;
    },
    addSelectTrajs: (state, action) => {
      const data = action.payload;
      if (!(state.selectTrajs.some(item => (item.id === data.id)))) {
        state.selectTrajs.push(data);
      }
    },
    delSelectTraj: (state, action) => {
      const id = action.payload;
      state.selectTrajs = state.selectTrajs.filter((item) => (item.id !== id))
    },
    addImgUrl2SelectTraj: (state, action) => {
      const { id, imgUrl } = action.payload;
      state.selectTrajs = state.selectTrajs.map(item => {
        if (item.id === id) {
          item.imgUrl = imgUrl;
        }
        return item;
      });
    },
    setCurShowTrajId: (state, action) => {
      state.curShowTrajId = action.payload;
    },
  }
})

export const {
  setBarData,
  setCalendarData,
  addSelectTrajs,
  delSelectTraj,
  addImgUrl2SelectTraj,
  setCurShowTrajId,
} = analysisReducer.actions;

export default analysisReducer.reducer;