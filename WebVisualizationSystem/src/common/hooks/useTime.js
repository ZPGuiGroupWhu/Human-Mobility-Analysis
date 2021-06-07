import { useReducer } from 'react'

// 存储用于筛选的时间信息
function reducer(state, action) {
  switch (action.type) {
    case 'dateStart':
      return {
        ...state,
        dateStart: action.payload,
      }
    case 'dateEnd':
      return {
        ...state,
        dateEnd: action.payload,
      }
    case 'hourStart':
      return {
        ...state,
        hourStart: action.payload,
      }
    case 'hourEnd':
      return {
        ...state,
        hourEnd: action.payload,
      }
  }
}

export const useTime = () => {
  /**
   * dateStart: 起始日期
   * dateEnd: 终止日期
   * hourStart: 起始时间
   * hourEnd: 终止时间
   */
  const initState = {
    dateStart: '',
    dateEnd: '',
    hourStart: '',
    hourEnd: '',
  }
  const [state, dispatch] = useReducer(reducer, initState);
  return [state, dispatch]
}