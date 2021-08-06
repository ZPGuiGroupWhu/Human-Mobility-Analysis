import { useEffect, useReducer, useCallback } from 'react';
import axios from 'axios';
import transcoords from '@/common/func/transcoords';

export const useReqod = () => {
  // 存储 od 数据
  const reducer = useCallback(
    (state, action) => {
      const { type, payload } = action;
      switch (type) {
        case 'org':
          return {
            ...state,
            org: payload,
          };
        case 'dest':
          return {
            ...state,
            dest: payload,
          };
        default:
          return;
      }
    },
    [],
  )
  const [state, dispatch] = useReducer(reducer, {
    org: [],
    dest: [],
  })

  // 数据并发请求
  useEffect(() => {
    const instance = axios.create({
      baseURL: 'http://localhost:7001',
      timeout: 2000,
    });
    instance.interceptors.response.use(
      res => (res.data),
      err => console.log(err)
    );
    
    function getOrg () {
      return instance.get('/org');
    }
    function getDest () {
      return instance.get('/dest');
    }

    axios.all([getOrg(), getDest()]).then(
      axios.spread(function (org, dest) {
        let orgData = org.reduce((prev, {id, coord, count}) => {
          return [...prev, coord];
        }, [])
        let destData = dest.reduce((prev, {id, coord, count}) => {
          return [...prev, coord];
        }, [])
        dispatch({type: 'org', payload: transcoords(orgData)});
        dispatch({type: 'dest', payload: transcoords(destData)});
      })
    )
  }, [])

  return state;
}