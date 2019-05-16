import { TestBed } from '@angular/core/testing';

import { LoadService } from './load.service';

describe('LoadService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: LoadService = TestBed.get(LoadService);
    expect(service).toBeTruthy();
  });
});
