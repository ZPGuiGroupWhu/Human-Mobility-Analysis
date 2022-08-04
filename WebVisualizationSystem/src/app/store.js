import { configureStore } from '@reduxjs/toolkit';
import selectReducer from './slice/selectSlice';
import analysisSlice from './slice/analysisSlice';
import predictReducer from './slice/predictSlice';

// redux store
export default configureStore({
  reducer: {
    select: selectReducer,
    analysis: analysisSlice,
    predict: predictReducer,
  }
});