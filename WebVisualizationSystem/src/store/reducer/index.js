export function reducer(state, action) {
  const { type, payload } = action;
  switch (type) {
    case 'setSelectedTraj':
      return {
        ...state,
        selectedTraj: payload,
      }
    default:
      return;
  }
}