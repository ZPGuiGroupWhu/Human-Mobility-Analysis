import {createSlice} from '@reduxjs/toolkit';

const initialState = {
  selectedTraj: {}, // 选择的单轨迹对象
}

const predictReducer = createSlice({
  name: 'predict',
  initialState,
  reducers: {
    setSelectedTraj: (state, action) => {
      state.selectedTraj = action.payload;
    }
  }
})

export const {setSelectedTraj} = predictReducer.actions;

export default predictReducer.reducer;