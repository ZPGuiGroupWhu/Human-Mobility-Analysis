export function reducer(state, action) {
  const { type, payload } = action;
  switch (type) {
    // select 页面
    case 'setAllData':
      return {
        ...state,
        allData: payload,
      }
    case 'setSelectedUsers':
      return {
        ...state,
        selectedUsers: payload,
      }
    // predict 页面
    case 'setSelectedTraj':
      return {
        ...state,
        selectedTraj: payload,
      }
    default:
      return { ...state };
  }
}