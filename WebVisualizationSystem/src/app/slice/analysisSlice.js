import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  hourCount: [],
  monthCount: [],
  weekdayCount: [],
  selectTrajs: [],  // 添加到“轨迹列表”中的轨迹数据集合
}

const analysisReducer = createSlice({
  name: 'analysis',
  initialState,
  reducers: {
    setAll: (state, action) => {
      Object.keys(state).forEach((item, idx) => {
        state[item] = action.payload[item]
      })
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
    }
  }
})

export const {
  setAll,
  addSelectTrajs,
  delSelectTraj,
  addImgUrl2SelectTraj
} = analysisReducer.actions;

export default analysisReducer.reducer;