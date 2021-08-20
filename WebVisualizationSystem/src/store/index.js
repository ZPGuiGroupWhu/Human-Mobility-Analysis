import { useReducer, createContext, useMemo } from 'react';
import { initArg } from './state';
import { reducer } from './reducer';

const Store = createContext({});

export const wrappedWithRedux = WrappedComponent => () => {
  const [state, dispatch] = useReducer(reducer, initArg);
  const store = useMemo(() => ({ state, dispatch }), [state]);
  return (
    <Store.Provider value={store}>
      <WrappedComponent />
    </Store.Provider>
  )
}


export default Store;