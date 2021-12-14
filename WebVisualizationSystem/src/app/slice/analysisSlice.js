import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  hourCount: [],
  monthCount: [],
  weekdayCount: [],
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
  }
})

export const { setAll } = analysisReducer.actions;

export default analysisReducer.reducer;