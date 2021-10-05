import { configureStore } from '@reduxjs/toolkit';
import selectReducer from './slice/selectSlice';
import predictReducer from './slice/predictSlice';

// redux store
export default configureStore({
  reducer: {
    select: selectReducer,
    predict: predictReducer,
  }
});